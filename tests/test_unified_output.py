"""One terminal session must retain all worker outcomes and stay live."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from queue import Queue
from threading import Thread


def _environment(addopts: str = "") -> dict[str, str]:
    environment = dict(os.environ, PYTEST_ADDOPTS=addopts, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    environment.pop("PYTEST_PLUGINS", None)
    return environment


def _run(root: Path, *arguments: str, addopts: str = "", workers: int = 2) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rtest", "--pytest-functions", "--unified-output", "-n", str(workers), *arguments],
        cwd=root,
        env=_environment(addopts),
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestUnifiedOutput:
    def test_one_summary_preserves_outcomes_warnings_and_captured_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_outcomes.py").write_text(
                "import pytest, warnings\n"
                "@pytest.fixture\ndef broken(): raise RuntimeError('fixture failure')\n"
                "def test_pass(): warnings.warn('worker warning')\n"
                "def test_fail():\n    print('captured worker output')\n    assert False, 'assertion failure'\n"
                "@pytest.mark.skip(reason='skip reason')\ndef test_skip(): pass\n"
                "@pytest.mark.xfail(reason='expected failure')\ndef test_xfail(): assert False\n"
                "@pytest.mark.xfail(reason='unexpected success')\ndef test_xpass(): pass\n"
                "def test_error(broken): pass\n"
            )
            result = _run(root)
            assert result.returncode == 1, result.stdout + result.stderr
            assert result.stdout.count("test session starts") == 1
            assert "collected 6 items" in result.stdout
            for text in (
                "1 passed",
                "1 failed",
                "1 skipped",
                "1 xfailed",
                "1 xpassed",
                "1 error",
                "assertion failure",
                "fixture failure",
                "captured worker output",
                "worker warning",
            ):
                assert text in result.stdout, result.stdout + result.stderr
            assert "starting worker" not in result.stderr

    def test_filtered_empty_workers_and_all_empty_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "conftest.py").write_text(
                "import os\nfrom pathlib import Path\n"
                "def pytest_collection_modifyitems(items):\n"
                "    Path(f'collection-{os.getpid()}').write_text(str(len(items)))\n"
            )
            (root / "test_filter.py").write_text("def test_keep(): pass\ndef test_drop(): pass\n")
            result = _run(root, addopts="-k keep")
            assert result.returncode == 0, result.stdout + result.stderr
            assert "1 passed, 1 deselected" in result.stdout
            assert len(list(root.glob("collection-*"))) == 2
            quiet = _run(root, "--env", "PYTEST_ADDOPTS=-q -k keep")
            assert quiet.returncode == 0, quiet.stdout + quiet.stderr
            assert "1 passed, 1 deselected" in quiet.stdout
            assert "test session starts" not in quiet.stdout
            empty = _run(root, addopts="-k missing")
            assert empty.returncode == 5, empty.stdout + empty.stderr
            assert "2 deselected" in empty.stdout

    def test_collection_errors_and_usage_errors_remain_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_import.py").write_text("raise RuntimeError('collection failure')\ndef test_case(): pass\n")
            result = _run(root)
            assert result.returncode == 4, result.stdout + result.stderr
            assert "collection failure" in result.stdout
            invalid = _run(root, "--dist", "invalid")
            assert invalid.returncode != 0
            native = _run(root, "--runner=native")
            assert native.returncode == 4
            assert "requires the pytest runner" in native.stderr

    def test_serial_collection_only_and_empty_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = _run(root)
            assert empty.returncode == 0, empty.stdout + empty.stderr
            (root / "test_plain.py").write_text("def test_one(): pass\ndef test_two(): pass\n")
            serial = _run(root, workers=1)
            assert serial.returncode == 0, serial.stdout + serial.stderr
            assert "2 passed" in serial.stdout
            collected = _run(root, "--collect-only")
            assert collected.returncode == 0, collected.stdout + collected.stderr
            assert "test_plain.py::test_one" in collected.stdout
            assert "test session starts" not in collected.stdout

    def test_missing_worker_reports_cannot_be_successful(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test_exit.py").write_text("import os\ndef test_exit(): os._exit(0)\n")
            result = _run(root)
            assert result.returncode != 0
            assert "complete reporting metadata" in result.stdout

    def test_progress_arrives_before_a_blocked_worker_can_finish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root / "continue"
            (root / "test_stream.py").write_text(
                "import time\nfrom pathlib import Path\n"
                "def test_fast(): pass\n"
                "def test_wait():\n"
                "    deadline = time.monotonic() + 25\n"
                "    while not Path('continue').exists():\n"
                "        assert time.monotonic() < deadline\n"
                "        time.sleep(0.01)\n"
            )
            received: Queue[str] = Queue()
            with subprocess.Popen(
                [sys.executable, "-m", "rtest", "--pytest-functions", "--unified-output", "-n", "2"],
                cwd=root,
                env=_environment("-q"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ) as process:
                assert process.stdout is not None

                def read() -> None:
                    assert process.stdout is not None
                    while character := process.stdout.read(1):
                        received.put(character)

                reader = Thread(target=read, daemon=True)
                reader.start()
                try:
                    # Quiet pytest prints a dot for the completed test. The
                    # other worker cannot exit until the parent sees that dot.
                    while received.get(timeout=15) != ".":
                        pass
                    assert process.poll() is None
                finally:
                    gate.touch()
                    process.wait(timeout=15)
                    reader.join(timeout=5)
                assert process.returncode == 0
