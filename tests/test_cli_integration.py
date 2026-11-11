"""Integration tests for CLI functionality."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import TextIO

import rtest
from rtest.exit_code import ExitCodeValues


class TestPytestFunctionSelection:
    def test_parallel_output_arrives_before_workers_can_finish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "continue"
            (root / "test_progress.py").write_text(
                "import sys, time\nfrom pathlib import Path\n"
                + "\n".join(
                    f"def test_worker_{index}():\n"
                    f"    print('stdout:{index}')\n"
                    f"    print('stderr:{index}', file=sys.stderr)\n"
                    "    deadline = time.monotonic() + 25\n"
                    "    while not Path('continue').exists():\n"
                    "        assert time.monotonic() < deadline, 'parent did not receive live output'\n"
                    "        time.sleep(0.01)\n"
                    for index in range(2)
                )
            )
            environment = os.environ.copy()
            environment.pop("PYTEST_PLUGINS", None)
            environment.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="-s", PYTHONUNBUFFERED="0")
            messages: Queue[tuple[str, str]] = Queue()

            def read(stream: TextIO, channel: str) -> None:
                for line in stream:
                    messages.put((channel, line))

            with subprocess.Popen(
                [sys.executable, "-m", "rtest", "--pytest-functions", "--quiet-collection", "-n", "2"],
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as process:
                assert process.stdout is not None and process.stderr is not None
                readers = [
                    Thread(target=read, args=(process.stdout, "stdout"), daemon=True),
                    Thread(target=read, args=(process.stderr, "stderr"), daemon=True),
                ]
                for reader in readers:
                    reader.start()
                expected = {(channel, index) for channel in ("stdout", "stderr") for index in range(2)}
                try:
                    while expected:
                        channel, line = messages.get(timeout=15)
                        for marker in expected.copy():
                            if marker[0] == channel and f"{channel}:{marker[1]}" in line:
                                expected.remove(marker)
                    assert process.poll() is None
                finally:
                    gate.touch()
                    process.wait(timeout=15)
                    for reader in readers:
                        reader.join(timeout=5)
                assert process.returncode == 0

    def test_parallel_workers_have_isolated_cache_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = (
                "    assert cache.get('worker-value', None) is None\n"
                "    cache.set('worker-value', 'present')\n"
                "    Path(f'cache-directory-{os.getpid()}.txt').write_text(str(cache._cachedir))\n"
            )
            (root / "test_cases.py").write_text(
                "import os\nfrom pathlib import Path\n"
                + "\n".join(f"def test_worker_{index}(cache):\n{body}" for index in range(4))
            )
            result = self._run(root, "-n", "4")
            assert result.returncode == 0, result.stdout + result.stderr
            paths = [path.read_text() for path in root.glob("cache-directory-*.txt")]
            assert len(paths) == len(set(paths)) == 4
            assert all(not Path(path).exists() for path in paths)

    def _run(self, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("PYTEST_PLUGINS", None)
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        environment["PYTEST_ADDOPTS"] = ""
        return subprocess.run(
            [sys.executable, "-m", "rtest", "--pytest-functions", *args],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_runtime_ids_and_fixtures_execute_all_variants_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "conftest.py").write_text(
                "import pytest\n"
                "@pytest.fixture\n"
                "def token(): return 'fixture'\n"
                "def pytest_make_parametrize_id(config, val, argname): return f'runtime-{val}'\n"
            )
            (root / "test_cases.py").write_text(
                "from pathlib import Path\n"
                "import pytest\n"
                "@pytest.mark.parametrize('value', [1, 2, 3])\n"
                "def test_cases(value, token):\n"
                "    assert token == 'fixture'\n"
                "    with Path('calls.txt').open('a') as stream: stream.write(f'{value}\\n')\n"
            )
            collected = self._run(root, "--collect-only")
            assert collected.returncode == 0, collected.stdout + collected.stderr
            assert collected.stdout.count("  test_cases.py::test_cases\n") == 1
            executed = self._run(root, "-n", "2")
            assert executed.returncode == 0, executed.stdout + executed.stderr
            assert sorted((root / "calls.txt").read_text().splitlines()) == ["1", "2", "3"]

    def test_rejects_native_runner_and_explicit_parameter_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arguments in (("--runner", "native"), ("test_cases.py::test_cases[one]",)):
                result = self._run(root, *arguments)
                assert result.returncode == ExitCodeValues.USAGE_ERROR

    def test_quiet_collection_suppresses_selectors_but_keeps_parse_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "test_cases.py"
            path.write_text("def test_case(): pass\n")
            result = self._run(root, "--quiet-collection", "--collect-only")
            assert result.returncode == 0, result.stdout + result.stderr
            assert "test_cases.py::test_case" not in result.stdout
            path.write_text("def test_broken(:\n")
            failed = self._run(root, "--quiet-collection", "--collect-only")
            assert failed.returncode != 0
            assert "ERROR" in failed.stdout

    def test_large_function_lists_execute_through_argument_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_cases.py").write_text(
                "\n".join(f"def test_function_with_a_long_selector_{index}(): pass" for index in range(1000))
            )
            result = self._run(root, "-n", "1")
            assert result.returncode == 0, result.stdout + result.stderr
            assert "1000 passed" in result.stdout

    def test_marker_filtering_allows_empty_workers_and_preserves_globally_empty_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pytest.ini").write_text("[pytest]\nmarkers = selected: selected test\n")
            (root / "test_cases.py").write_text(
                "import pytest\n"
                "@pytest.mark.selected\n"
                "def test_selected(): pass\n"
                "def test_other_a(): pass\n"
                "def test_other_b(): pass\n"
                "def test_other_c(): pass\n"
            )
            selected = self._run(root, "-n", "4", "--env", "PYTEST_ADDOPTS=-m selected -q")
            assert selected.returncode == 0, selected.stdout + selected.stderr
            assert "1 passed" in selected.stdout
            empty = self._run(root, "-n", "4", "--env", "PYTEST_ADDOPTS=-m absent -q")
            assert empty.returncode == 5, empty.stdout + empty.stderr


class TestCLIBasics:
    """Basic CLI tests."""

    def test_help_shows_usage(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "rtest", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == ExitCodeValues.OK
        assert "Usage:" in result.stdout
        assert "--runner" in result.stdout
        assert "--env" in result.stdout
        assert "-n" in result.stdout

    def test_version_shows_version(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "rtest", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == ExitCodeValues.OK
        assert "rtest" in result.stdout.lower()

    def test_invalid_flag_rejected(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "rtest", "--invalid-flag-xyz"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


class TestCLIErrorHandling:
    """Tests for error handling."""

    def test_nonexistent_file(self) -> None:
        """Nonexistent file should return exit code 4 (matching pytest behavior)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result = subprocess.run(
                [sys.executable, "-m", "rtest", "--collect-only", "nonexistent.py"],
                capture_output=True,
                text=True,
                cwd=str(tmp_path),
            )
            assert result.returncode == ExitCodeValues.USAGE_ERROR
            assert "file or directory not found" in result.stderr

    def test_invalid_dist_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result = subprocess.run(
                [sys.executable, "-m", "rtest", "--dist", "invalid_mode_xyz"],
                capture_output=True,
                text=True,
                cwd=str(tmp_path),
            )
            assert result.returncode != 0

    @rtest.mark.cases("runner", ["native", "pytest"])
    @rtest.mark.cases("option", ["-n", "--maxprocesses"])
    def test_zero_worker_configuration_rejected(self, runner: str, option: str) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "test_failure.py").write_text("def test_failure(): assert False\n")
            result = subprocess.run(
                [sys.executable, "-m", "rtest", "--runner", runner, option, "0"],
                cwd=root,
                capture_output=True,
                text=True,
            )
            assert result.returncode != 0
            assert "invalid value" in result.stderr


class TestNativeRunnerEndToEnd:
    """End-to-end tests for native runner."""

    def test_native_runner_basic_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test_example.py"
            test_file.write_text("def test_pass(): assert True\ndef test_fail(): assert False\n")

            result = subprocess.run(
                [sys.executable, "-m", "rtest", "--runner", "native", "-n", "1"],
                capture_output=True,
                text=True,
                cwd=str(tmp_path),
            )
            assert result.returncode == ExitCodeValues.TESTS_FAILED
            assert "1 passed" in result.stdout
            assert "1 failed" in result.stdout
            assert "FAILED" in result.stdout

    def test_native_runner_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            result = subprocess.run(
                [sys.executable, "-m", "rtest", "--runner", "native"],
                capture_output=True,
                text=True,
                cwd=str(tmp_path),
            )
            assert result.returncode == ExitCodeValues.OK
            assert "No tests found" in result.stdout

    def test_native_runner_import_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            test_file = tmp_path / "test_bad.py"
            test_file.write_text("import nonexistent_module_xyz_abc\n\ndef test_never_runs(): pass\n")

            result = subprocess.run(
                [sys.executable, "-m", "rtest", "--runner", "native", "-n", "1"],
                capture_output=True,
                text=True,
                cwd=str(tmp_path),
            )
            assert result.returncode == ExitCodeValues.TESTS_FAILED
            assert "ModuleNotFoundError" in result.stdout

    @rtest.mark.cases(
        "source",
        [
            "import os\ndef test_exit(): os._exit(0)\n",
            "from rtest.worker.runner import TestResult\n"
            "TestResult.to_json = lambda self: '{invalid'\n"
            "def test_pass(): pass\n",
        ],
    )
    def test_missing_or_malformed_worker_results_fail_the_run(self, source: str) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "test_worker.py").write_text(source)
            result = subprocess.run(
                [sys.executable, "-m", "rtest", "--runner", "native"],
                cwd=root,
                capture_output=True,
                text=True,
            )
            assert result.returncode == ExitCodeValues.TESTS_FAILED
            assert "FAILED" in result.stdout
            assert "1 worker error" in result.stdout


class TestPytestEnvironment:
    """Environment options reach both sequential and parallel pytest processes."""

    @rtest.mark.cases("workers", [1, 2])
    def test_environment_values_and_invalid_option_warning(self, workers: int) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "test_env.py").write_text(
                "import os\n"
                "def test_value(): assert os.environ['RTEST_VALUE'] == 'a=b'\n"
                "def test_empty(): assert os.environ['RTEST_EMPTY'] == ''\n"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rtest",
                    "--runner",
                    "pytest",
                    "-n",
                    str(workers),
                    "--env",
                    "RTEST_VALUE=old",
                    "--env",
                    "RTEST_VALUE=a=b",
                    "--env",
                    "RTEST_EMPTY=",
                    "--env",
                    "INVALID_NO_EQUALS",
                ],
                capture_output=True,
                text=True,
                cwd=tmp_path,
            )
            assert result.returncode == ExitCodeValues.OK, result.stdout + result.stderr
            assert result.stderr.count("Warning: Invalid environment variable format: INVALID_NO_EQUALS") == 1
