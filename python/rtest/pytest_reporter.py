"""Forward worker reports to the unified controller without a worker terminal."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TextIO

import pytest

REPORT_DIRECTORY = "_RTEST_REPORT_DIRECTORY"
_stream: TextIO | None = None
_deselected = 0


def _emit(kind: str, **data: Any) -> None:
    global _stream
    if _stream is None:
        path = Path(os.environ[REPORT_DIRECTORY]) / f"worker-{os.getpid()}.jsonl"
        _stream = path.open("a", encoding="utf-8", buffering=1)
    _stream.write(json.dumps({"kind": kind, **data}) + "\n")


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    reporter = config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        config.pluginmanager.unregister(reporter)


def pytest_collection_finish(session: pytest.Session) -> None:
    _emit("collection", selected=len(session.items), deselected=_deselected)


def pytest_deselected(items: list[pytest.Item]) -> None:
    global _deselected
    _deselected += len(items)


@pytest.hookimpl
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    _emit("report", data=report._to_json())


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.failed or report.skipped:
        _emit("report", data=report._to_json())


def pytest_warning_recorded(warning_message: Any, when: str, nodeid: str) -> None:
    _emit(
        "warning",
        message=str(warning_message.message),
        category=warning_message.category.__name__,
        filename=warning_message.filename,
        lineno=warning_message.lineno,
        when=when,
        nodeid=nodeid,
    )


def pytest_sessionfinish(exitstatus: int) -> None:
    _emit("finish", exitstatus=int(exitstatus))
