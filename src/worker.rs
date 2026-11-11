use std::fmt;
use std::path::PathBuf;
use std::process::ExitStatus;
use std::thread;

use crate::pytest_executor::pytest_command;

#[derive(Debug)]
pub enum WorkerError {
    Launch(std::io::Error),
    Panicked,
}

impl fmt::Display for WorkerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Launch(error) => write!(f, "Failed to execute command: {error}"),
            Self::Panicked => write!(f, "Worker thread panicked"),
        }
    }
}

impl std::error::Error for WorkerError {}

#[derive(Debug)]
pub struct WorkerResult {
    pub worker_id: usize,
    pub result: Result<ExitStatus, WorkerError>,
}

/// Configuration for a worker task
pub struct WorkerTask {
    pub worker_id: usize,
    pub program: String,
    pub initial_args: Vec<String>,
    pub tests: Vec<String>,
    pub working_dir: PathBuf,
    pub env_vars: Vec<(String, String)>,
}

#[derive(Default)]
pub struct WorkerPool {
    workers: Vec<WorkerHandle>,
}

struct WorkerHandle {
    id: usize,
    handle: thread::JoinHandle<std::io::Result<ExitStatus>>,
}

impl WorkerPool {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spawn_worker(&mut self, task: WorkerTask) {
        let worker_id = task.worker_id;
        if std::env::var_os("_RTEST_REPORT_DIRECTORY").is_none() {
            eprintln!(
                "rtest: starting worker {worker_id} ({} selectors)",
                task.tests.len()
            );
        }
        let handle = thread::spawn(move || {
            // Independent pytest sessions cannot atomically share cache updates
            // on every supported platform, notably Windows.
            let cache = tempfile::tempdir()?;
            let pytest_args = vec![
                "-o".to_string(),
                format!("cache_dir={}", cache.path().display()),
            ];
            let mut prepared = pytest_command(
                &task.program,
                &task.initial_args,
                &task.tests,
                &pytest_args,
                Some(&task.working_dir),
                &task.env_vars,
            )?;
            // Inherit both streams instead of retaining all progress until every
            // worker exits. Python must also flush when rtest itself is piped.
            prepared.command.env("PYTHONUNBUFFERED", "1").status()
        });

        self.workers.push(WorkerHandle {
            id: worker_id,
            handle,
        });
    }

    pub fn wait_for_all(self) -> Vec<WorkerResult> {
        let mut results: Vec<_> = self
            .workers
            .into_iter()
            .map(|worker| WorkerResult {
                worker_id: worker.id,
                result: worker
                    .handle
                    .join()
                    .map_err(|_| WorkerError::Panicked)
                    .and_then(|result| result.map_err(WorkerError::Launch)),
            })
            .collect();
        results.sort_by_key(|result| result.worker_id);
        results
    }
}
