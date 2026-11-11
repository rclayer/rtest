use crate::{subproject, DistributionMode, WorkerPool, WorkerTask};
use std::num::NonZeroUsize;
use std::path::Path;

pub struct PytestRunner {
    pub program: String,
    pub initial_args: Vec<String>,
    pub env_vars: Vec<(String, String)>,
}

/// Empty filtered shards are harmless when another shard selected tests.
pub(crate) fn pytest_exit_code(codes: impl IntoIterator<Item = i32>) -> i32 {
    let mut success = false;
    let mut empty = false;
    let mut failure = 0;
    for code in codes {
        match code {
            0 => success = true,
            5 => empty = true,
            _ => failure = code,
        }
    }
    if failure != 0 {
        failure
    } else if empty && !success {
        5
    } else {
        0
    }
}

/// Configuration for parallel test execution
pub struct ParallelExecutionConfig<'a> {
    pub program: &'a str,
    pub initial_args: &'a [String],
    pub worker_count: NonZeroUsize,
    pub dist_mode: DistributionMode,
    pub rootpath: &'a Path,
    pub env_vars: &'a [(String, String)],
}

impl PytestRunner {
    pub fn new(program: String, env_vars: &[String]) -> Self {
        let mut initial_args = vec!["-m".into(), "pytest".into()];
        if std::env::var_os("_RTEST_REPORT_DIRECTORY").is_some() {
            initial_args.extend([
                "-p".into(),
                "no:sugar".into(),
                "-p".into(),
                "rtest.pytest_reporter".into(),
            ]);
        }

        // Parse environment variables from KEY=VALUE format
        let parsed_env_vars = parse_env_vars(env_vars, |env_str| {
            eprintln!("Warning: Invalid environment variable format: {}", env_str);
        });

        PytestRunner {
            program,
            initial_args,
            env_vars: parsed_env_vars,
        }
    }
}

pub(crate) fn report_worker_count(count: usize) -> std::io::Result<()> {
    if let Some(directory) = std::env::var_os("_RTEST_REPORT_DIRECTORY") {
        std::fs::write(
            Path::new(&directory).join("runner.jsonl"),
            format!("{{\"kind\":\"workers\",\"count\":{count}}}\n"),
        )?;
    }
    Ok(())
}

pub(crate) fn parse_env_vars(env_vars: &[String], warn: impl Fn(&str)) -> Vec<(String, String)> {
    env_vars
        .iter()
        .filter_map(|env_str| match env_str.split_once('=') {
            Some((key, value)) => Some((key.into(), value.into())),
            None => {
                warn(env_str);
                None
            }
        })
        .collect()
}

/// Execute tests in parallel across multiple workers
pub fn execute_tests_parallel(config: &ParallelExecutionConfig, test_nodes: Vec<String>) -> i32 {
    log::info!(
        "Running tests with {} workers using {} distribution",
        config.worker_count,
        config.dist_mode
    );

    let test_groups = subproject::group_tests_by_subproject(config.rootpath, test_nodes);
    let mut worker_pool = WorkerPool::new();
    let mut worker_id = 0;

    for (subproject_root, tests) in test_groups {
        for tests in config
            .dist_mode
            .distribute_tests(tests, config.worker_count)
        {
            worker_pool.spawn_worker(WorkerTask {
                worker_id,
                program: config.program.to_string(),
                initial_args: config.initial_args.to_vec(),
                tests,
                working_dir: subproject_root.clone(),
                env_vars: config.env_vars.to_vec(),
            });
            worker_id += 1;
        }
    }

    let report_error = report_worker_count(worker_id).err();
    if let Some(error) = &report_error {
        eprintln!("Cannot report worker count: {error}");
    }
    let mut exit_codes = Vec::new();
    for result in worker_pool.wait_for_all() {
        let code = match result.result {
            Ok(status) => status.code().unwrap_or_else(|| {
                eprintln!("Worker {} terminated: {status}", result.worker_id);
                1
            }),
            Err(error) => {
                eprintln!("{error}");
                1
            }
        };
        exit_codes.push(code);
    }

    if report_error.is_some() {
        1
    } else {
        pytest_exit_code(exit_codes)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn empty_shards_preserve_success_failure_and_globally_empty_selections() {
        for (codes, expected) in [
            (vec![0, 5, 5], 0),
            (vec![5, 0, 5], 0),
            (vec![5, 5], 5),
            (vec![1, 5, 0], 1),
            (vec![4, 5], 4),
            (vec![], 0),
        ] {
            assert_eq!(super::pytest_exit_code(codes), expected);
        }
    }

    #[test]
    fn test_execute_tests_nonexistent_program() {
        let exit_code = crate::pytest_executor::execute_tests(
            "nonexistent_program_xyz_12345",
            &[],
            vec!["test::something".into()],
            vec![],
            None,
            &[],
        );
        assert_eq!(exit_code, 1);
    }
}
