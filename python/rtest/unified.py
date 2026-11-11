"""Replay worker reports through one pytest session, without parent collection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import pytest
from _pytest.terminal import TerminalReporter

from rtest.pytest_reporter import REPORT_DIRECTORY


class _Controller:
    def __init__(self, arguments: list[str], directory: Path, environment: dict[str, str]) -> None:
        self.arguments = arguments
        self.directory = directory
        self.environment = environment
        self.offsets: dict[Path, int] = {}
        self.expected: int | None = None
        self.collected: dict[Path, tuple[int, int]] = {}
        self.finished: set[Path] = set()
        self.pending: list[dict[str, Any]] = []
        self.collection_finished = False
        self.exitstatus: int | None = None

    @pytest.hookimpl(tryfirst=True)
    def pytest_configure(self, config: pytest.Config) -> None:
        # The reporter is serial; only the Rust subprocess launches workers.
        if hasattr(config.option, "numprocesses"):
            config.option.numprocesses = 0

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection(self) -> bool:
        return True

    def _read_events(self) -> list[tuple[Path, dict[str, Any]]]:
        events = []
        for path in sorted(self.directory.glob("*.jsonl")):
            with path.open(encoding="utf-8") as stream:
                stream.seek(self.offsets.get(path, 0))
                while line := stream.readline():
                    if not line.endswith("\n"):
                        break
                    events.append((path, json.loads(line)))
                    self.offsets[path] = stream.tell()
        return events

    def _dispatch(self, config: pytest.Config, event: dict[str, Any]) -> None:
        if event["kind"] == "report":
            data = event["data"]
            if "when" in data:
                config.hook.pytest_runtest_logreport(report=pytest.TestReport._from_json(data))
            else:
                config.hook.pytest_collectreport(report=pytest.CollectReport._from_json(data))
        elif event["kind"] == "warning":
            category = type(event["category"], (Warning,), {})
            warning = warnings.WarningMessage(category(event["message"]), category, event["filename"], event["lineno"])
            config.hook.pytest_warning_recorded.call_historic(
                kwargs={"warning_message": warning, "when": event["when"], "nodeid": event["nodeid"], "location": None}
            )

    def _drain(self, session: pytest.Session, *, final: bool = False) -> None:
        for path, event in self._read_events():
            kind = event["kind"]
            if kind == "workers":
                self.expected = event["count"]
                reporter = session.config.pluginmanager.getplugin("terminalreporter")
                if isinstance(reporter, TerminalReporter):
                    reporter.write_line(f"rtest: {self.expected} workers collecting")
            elif kind == "collection":
                self.collected[path] = event["selected"], event["deselected"]
                reporter = session.config.pluginmanager.getplugin("terminalreporter")
                if isinstance(reporter, TerminalReporter) and reporter.isatty() and self.expected is not None:
                    reporter.rewrite(f"collecting: {len(self.collected)}/{self.expected} workers ready")
            elif kind == "finish":
                self.finished.add(path)
            elif self.collection_finished:
                self._dispatch(session.config, event)
            else:
                self.pending.append(event)
        if not self.collection_finished and (final or self.expected == len(self.collected)):
            self.collection_finished = True
            selected = sum(count[0] for count in self.collected.values())
            deselected = sum(count[1] for count in self.collected.values())
            session.testscollected = selected
            reporter = session.config.pluginmanager.getplugin("terminalreporter")
            if isinstance(reporter, TerminalReporter):
                # No parent Collectors/Items exist; only their aggregate counts
                # belong in the terminal report. These fields are pytest's totals.
                reporter._numcollected = selected + deselected
                reporter.stats.setdefault("deselected", []).extend([None] * deselected)
            session.config.hook.pytest_collection_finish(session=session)
            for event in self.pending:
                self._dispatch(session.config, event)
            self.pending.clear()

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtestloop(self, session: pytest.Session) -> bool:
        arguments = self.arguments.copy()
        if "--quiet-collection" not in arguments:
            arguments.append("--quiet-collection")
        # Include environment initialized by the parent's pytest configuration
        # (for example a shared random seed), then apply explicit CLI overrides.
        environment = dict(os.environ, **self.environment)
        with subprocess.Popen([sys.executable, "-m", "rtest", *arguments], env=environment) as process:
            try:
                while process.poll() is None:
                    self._drain(session)
                    time.sleep(0.025)
                self._drain(session, final=True)
                if self.expected is None and not self.collected and process.returncode == 0:
                    self.expected = 0
                if process.returncode == 0 and len(self.finished) != self.expected:
                    raise RuntimeError("rtest workers finished without complete reporting metadata")
                self.exitstatus = process.returncode if process.returncode >= 0 else 2
            finally:
                if process.poll() is None:
                    process.terminate()
        return True

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        if self.exitstatus is not None:
            session.exitstatus = pytest.ExitCode(self.exitstatus)


def run(arguments: list[str]) -> int:
    """Run one reporting-only pytest session around the Rust worker launcher."""
    if any(
        value == "--runner=native" or (value == "--runner" and arguments[index + 1 : index + 2] == ["native"])
        for index, value in enumerate(arguments)
    ):
        print("--unified-output requires the pytest runner", file=sys.stderr)
        return 4
    with tempfile.TemporaryDirectory(prefix="rtest-reports-") as directory:
        root = Path(directory)
        environment = {REPORT_DIRECTORY: directory}
        for index, argument in enumerate(arguments):
            assignment = (
                arguments[index + 1]
                if argument in {"-e", "--env"} and index + 1 < len(arguments)
                else argument.removeprefix("--env=")
                if argument.startswith("--env=")
                else ""
            )
            if "=" in assignment:
                key, value = assignment.split("=", 1)
                environment[key] = value
        controller = _Controller(arguments, root, environment)
        previous_addopts = os.environ.get("PYTEST_ADDOPTS")
        if "PYTEST_ADDOPTS" in environment:
            os.environ["PYTEST_ADDOPTS"] = environment["PYTEST_ADDOPTS"]
        try:
            return int(
                pytest.main(
                    ["-o", "addopts=", "-o", f"cache_dir={root / 'cache'}", "-p", "no:sugar"], plugins=[controller]
                )
            )
        finally:
            if previous_addopts is None:
                os.environ.pop("PYTEST_ADDOPTS", None)
            else:
                os.environ["PYTEST_ADDOPTS"] = previous_addopts
