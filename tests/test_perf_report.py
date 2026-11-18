"""Harness tests for the opt-in runtime metrics (tests/perf_report.py) and the
pytest-lifetime wrapper (scripts/run_test_benchmark.py).

Behavior that needs a real pytest session runs a small synthetic suite in a child
pytest process: a temp directory with its own ``pytest.ini`` (so the repository's
``addopts`` and conftest stay out of it) and a conftest wired exactly like
``tests/conftest.py``. Summary arithmetic and partial-file reading are tested on
hand-written event files instead, which is cheaper and states the numbers.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import perf_report
import pytest

from bmad_loop.process_host import get_process_host

TESTS_DIR = Path(__file__).resolve().parent
WRAPPER = TESTS_DIR.parent / "scripts" / "run_test_benchmark.py"
sys.path.insert(0, str(WRAPPER.parent))
import run_test_benchmark  # noqa: E402

# Unique per session: a match anywhere in a metrics file can only be a leak.
CANARY_ENV = f"canary-env-{uuid.uuid4().hex}"
CANARY_OUT = f"canary-out-{uuid.uuid4().hex}"
CANARY_ARGV = f"canary-argv-{uuid.uuid4().hex}"
CANARY_FAIL = f"canary-fail-{uuid.uuid4().hex}"

SYNTH_CONFTEST = f"""\
import sys
sys.path.insert(0, {str(TESTS_DIR)!r})
import perf_report


def pytest_addoption(parser):
    perf_report.add_options(parser)


def pytest_configure(config):
    perf_report.configure(config)
"""

# Bound on any one synthetic child run; a hang fails the test instead of the job.
CHILD_TIMEOUT_S = 180


def _child_env(**extra: str) -> dict[str, str]:
    # The outer run's PYTEST_* variables (xdist worker identity, addopts) must not
    # leak into the synthetic session.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env.update(extra)
    return env


def _suite(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\naddopts = -p no:cacheprovider\n", encoding="utf-8")
    (root / "conftest.py").write_text(SYNTH_CONFTEST, encoding="utf-8")
    for name, body in files.items():
        (root / name).write_text(body, encoding="utf-8")
    return root


def _run(argv: list[str], cwd: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # fixed argv, no shell
        argv,
        cwd=cwd,
        env=_child_env(**env),
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
        check=False,
    )


def _pytest(suite: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return _run([sys.executable, "-m", "pytest", *args], suite, **env)


def _wrapped(
    suite: Path, metrics: Path, *pytest_args: str, wrapper_args: tuple[str, ...] = (), **env: str
) -> subprocess.CompletedProcess[str]:
    argv = [
        sys.executable,
        str(WRAPPER),
        "--metrics-dir",
        str(metrics),
        *wrapper_args,
        "--",
        *pytest_args,
    ]
    return _run(argv, suite, **env)


def _output(proc: subprocess.CompletedProcess[str]) -> str:
    return f"rc={proc.returncode}\n--- stdout\n{proc.stdout[-4000:]}\n--- stderr\n{proc.stderr[-4000:]}"


def _events(metrics: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for path in sorted(metrics.glob("events-*.jsonl")):
        records, malformed = perf_report.read_events(path)
        assert malformed == 0, path
        out[path.name] = records
    return out


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------ a real xdist run, shared read-only

MIXED_SUITE = """\
import os
import subprocess
import sys

import pytest


def test_passes():
    pass


def test_prints_and_spawns():
    print(os.environ["METRICS_CANARY_OUT"])
    print(os.environ["METRICS_CANARY_OUT"], file=sys.stderr)
    subprocess.run(
        [sys.executable, "-c", "import sys; print(sys.argv[1])", os.environ["METRICS_CANARY_ARGV"]],
        check=True,
    )
    subprocess.run(["git", "-c", "metrics.canary=" + os.environ["METRICS_CANARY_ARGV"], "version"], check=True)


def test_fails():
    assert os.environ["METRICS_CANARY_FAIL"] == "", "fails on purpose"


@pytest.mark.skip(reason="static skip")
def test_skipped():
    pass


def test_deselected_beta():
    pass


@pytest.mark.xdist_group("grouped")
def test_grouped():
    pass


@pytest.mark.parametrize("n", range(6))
def test_many(n):
    pass
"""


@pytest.fixture(scope="module")
def xdist_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    """One two-worker loadgroup run through the wrapper, with a failure, a skip, a
    deselection, a grouped test, printed output and subprocess attribution on."""
    base = tmp_path_factory.mktemp("xdist-run")
    suite = _suite(base / "suite", {"test_mixed.py": MIXED_SUITE})
    metrics = base / "metrics"
    proc = _wrapped(
        suite,
        metrics,
        "-q",
        "-rP",  # print passed tests' captured output, so the canary provably reaches pytest
        "-n",
        "2",
        "--dist",
        "loadgroup",
        "-k",
        "not deselected_beta",
        "--test-metrics-subprocesses",
        METRICS_CANARY_ENV=CANARY_ENV,
        METRICS_CANARY_OUT=CANARY_OUT,
        METRICS_CANARY_ARGV=CANARY_ARGV,
        METRICS_CANARY_FAIL=CANARY_FAIL,
    )
    return proc, metrics, suite


def test_wrapper_preserves_a_failed_runs_exit_status(xdist_run):
    proc, metrics, _ = xdist_run
    assert proc.returncode == 1, _output(proc)
    runner = _json(metrics / perf_report.RUNNER)
    assert (runner["state"], runner["exit_code"]) == ("exited", 1)
    summary = _json(metrics / perf_report.SUMMARY_JSON)
    assert (summary["status"], summary["exitstatus"], summary["runner_exit_code"]) == (
        "tests_failed",
        1,
        1,
    )
    assert summary["tests"]["outcomes"] == {"failed": 1, "passed": 9, "skipped": 1}


def test_worker_files_do_not_collide(xdist_run):
    proc, metrics, _ = xdist_run
    events = _events(metrics)
    starts = {name: records[0] for name, records in events.items()}
    assert all(record["kind"] == "session_start" for record in starts.values())
    # Each file is named for, and only written by, the process that opened it.
    for name, record in starts.items():
        assert name == f"events-{record['worker']}-{record['pid']}.jsonl"
    roles = sorted(record["role"] for record in starts.values())
    assert roles == ["controller", "worker", "worker"], _output(proc)
    assert len({(r["worker"], r["pid"]) for r in starts.values()}) == len(starts)

    # Every selected test ran on exactly one worker; no record is duplicated.
    executed: list[str] = []
    for records in events.values():
        executed.extend(r["nodeid"] for r in records if r["kind"] == "start")
    selected = _json(next(metrics.glob("inventory-gw0-*.json")))["selected"]
    assert sorted(executed) == sorted(selected)
    assert len(executed) == len(set(executed)) == 11


def test_collected_and_selected_inventories_are_distinct(xdist_run):
    _, metrics, _ = xdist_run
    inventories = {
        path.name.split("-")[1]: _json(path) for path in metrics.glob("inventory-*.json")
    }
    assert sorted(inventories) == ["gw0", "gw1"]  # both workers collect; the controller does not
    first = inventories["gw0"]
    # One full copy; the other worker's digests must match it.
    assert "selected" not in inventories["gw1"]
    assert inventories["gw1"]["digests"] == first["digests"]
    assert first["digests"]["selected"] == perf_report.nodeid_digest(first["selected"])
    assert first["digests"]["collected"] != first["digests"]["selected"]
    beta = "test_mixed.py::test_deselected_beta"
    assert beta in first["collected"]
    assert beta not in first["selected"]
    assert first["deselected"] == [beta]
    assert len(first["collected"]) == len(first["selected"]) + 1
    # loadgroup's worker-only `@group` suffix is stripped, so inventories compare
    # equal with and without -n.
    assert "test_mixed.py::test_grouped" in first["selected"]
    assert not any("@" in nodeid for nodeid in first["collected"])
    summary = _json(metrics / perf_report.SUMMARY_JSON)["tests"]
    assert (summary["collected"], summary["selected"], summary["deselected"]) == (12, 11, 1)
    assert summary["consistent"] is True
    assert (summary["started"], summary["not_started"], summary["unfinished"]) == (11, 0, [])


def test_no_raw_environment_or_child_output_is_recorded(xdist_run):
    proc, metrics, suite = xdist_run
    # The canaries did reach the child's output and the run's subprocesses ...
    assert CANARY_OUT in proc.stdout + proc.stderr, _output(proc)
    summary = _json(metrics / perf_report.SUMMARY_JSON)
    spawns = summary["subprocesses"]
    assert spawns["by_command"].get("git version") == 1
    assert spawns["by_command"].get("python", 0) >= 1  # the controller also launches its workers
    assert spawns["by_test"] == {"test_mixed.py::test_prints_and_spawns": 2}
    # ... and none of them, nor the machine-specific suite path, reached a file.
    files = sorted(p for p in metrics.iterdir() if p.is_file())
    assert {p.name for p in files} >= {
        perf_report.MANIFEST,
        perf_report.RUNNER,
        perf_report.SUMMARY_JSON,
    }
    for path in files:
        text = path.read_text(encoding="utf-8")
        for secret in (CANARY_ENV, CANARY_OUT, CANARY_ARGV, CANARY_FAIL, str(suite)):
            assert secret not in text, f"{secret!r} leaked into {path.name}"


def test_summary_reports_both_clocks_for_a_real_run(xdist_run):
    _, metrics, _ = xdist_run
    summary = _json(metrics / perf_report.SUMMARY_JSON)
    assert summary["elapsed"]["session_seconds"] > 0
    assert summary["elapsed"]["process_seconds"] >= summary["elapsed"]["session_seconds"]
    assert set(summary["worker_seconds"]["by_worker"]) == {"gw0", "gw1"}
    manifest = summary["manifest"]
    assert (manifest["numprocesses"], manifest["dist"], manifest["subprocess_attribution"]) == (
        2,
        "loadgroup",
        True,
    )
    assert manifest["selection"] == {
        "args": ["."],
        "ignore": [],
        "keyword_expression": True,
        "mark_expression": False,
    }


# ------------------------------------------------------------- the default path


INERT_SUITE = """\
import json
import os
from pathlib import Path


def test_reports_registration(pytestconfig):
    Path(os.environ["METRICS_PROBE_OUT"]).write_text(json.dumps({
        "registered": pytestconfig.pluginmanager.get_plugin("bmad-loop-test-metrics") is not None,
        "option": pytestconfig.getoption("test_metrics_dir"),
    }), encoding="utf-8")
"""


def test_without_the_option_nothing_is_registered_or_written(tmp_path):
    suite = _suite(tmp_path / "suite", {"test_inert.py": INERT_SUITE})
    probe = tmp_path / "probe.json"
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}

    plain = _pytest(suite, "-q", METRICS_PROBE_OUT=str(probe))
    assert plain.returncode == 0, _output(plain)
    assert _json(probe) == {"registered": False, "option": None}
    created = (
        {p.relative_to(tmp_path) for p in tmp_path.rglob("*")}
        - before
        - {probe.relative_to(tmp_path)}
    )
    assert {p for p in created if "__pycache__" not in p.parts} == set()

    # The same test sees the plugin when the option is given (through the
    # wrapper, whose exit status is the passing run's 0) — so the check above
    # can tell the two apart.
    wrapped = _wrapped(suite, tmp_path / "metrics", "-q", METRICS_PROBE_OUT=str(probe))
    assert wrapped.returncode == 0, _output(wrapped)
    assert _json(probe)["registered"] is True
    assert _json(tmp_path / "metrics" / perf_report.RUNNER)["exit_code"] == 0
    assert _json(tmp_path / "metrics" / perf_report.SUMMARY_JSON)["status"] == "passed"


def test_this_suite_registers_the_options(pytestconfig):
    # Raises ValueError if tests/conftest.py stopped registering it.
    pytestconfig.getoption("test_metrics_dir")
    pytestconfig.getoption("test_metrics_subprocesses")


def test_usage_errors_keep_pytest_exit_status(tmp_path):
    suite = _suite(tmp_path / "suite", {"test_inert.py": "def test_ok():\n    pass\n"})

    orphan_flag = _pytest(suite, "-q", "--test-metrics-subprocesses")
    assert orphan_flag.returncode == 4, _output(orphan_flag)
    assert "requires --test-metrics-dir" in orphan_flag.stderr

    used = tmp_path / "used"
    used.mkdir()
    (used / perf_report.MANIFEST).write_text("{}", encoding="utf-8")
    reuse = _pytest(suite, "-q", f"--test-metrics-dir={used}")
    assert reuse.returncode == 4, _output(reuse)
    assert "already holds a metrics run" in reuse.stderr
    assert sorted(p.name for p in used.iterdir()) == [perf_report.MANIFEST]

    bad_arg = _wrapped(suite, tmp_path / "metrics", "-q", "--no-such-pytest-flag")
    assert bad_arg.returncode == 4, _output(bad_arg)
    runner = _json(tmp_path / "metrics" / perf_report.RUNNER)
    assert (runner["state"], runner["exit_code"]) == ("exited", 4)
    assert _json(tmp_path / "metrics" / perf_report.SUMMARY_JSON)["status"] == "usage_error"


# ------------------------------------------------------------ summary arithmetic


def _write_events(path: Path, records: list[dict[str, Any]], tail: str = "") -> None:
    lines = [json.dumps({"v": 1, **r}) for r in records]
    path.write_text("\n".join(lines) + "\n" + tail, encoding="utf-8")


def _worker_records(name: str, pid: int, tests: list[tuple[str, float]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {"kind": "session_start", "mono": 100.0, "worker": name, "role": "worker", "pid": pid}
    ]
    mono = 100.0
    for nodeid, seconds in tests:
        records.append({"kind": "start", "mono": mono, "nodeid": nodeid})
        for when, duration in (("setup", 0.0), ("call", seconds), ("teardown", 0.0)):
            mono += duration
            records.append(
                {
                    "kind": "phase",
                    "mono": mono,
                    "nodeid": nodeid,
                    "when": when,
                    "outcome": "passed",
                    "duration": duration,
                }
            )
    return records


def test_summary_does_not_conflate_worker_seconds_with_elapsed(tmp_path):
    # Two workers, each busy for 2s over the same 2s of wall time.
    _write_events(
        tmp_path / "events-controller-1.jsonl",
        [
            {
                "kind": "session_start",
                "mono": 100.0,
                "worker": "controller",
                "role": "controller",
                "pid": 1,
            },
            {"kind": "sessionfinish_start", "mono": 102.0, "exitstatus": 0},
            {"kind": "session_finish", "mono": 102.0, "exitstatus": 0},
        ],
    )
    _write_events(
        tmp_path / "events-gw0-2.jsonl",
        _worker_records("gw0", 2, [("a.py::t1", 1.0), ("a.py::t2", 1.0)]),
    )
    _write_events(
        tmp_path / "events-gw1-3.jsonl",
        _worker_records("gw1", 3, [("b.py::t3", 1.0), ("b.py::t4", 1.0)]),
    )

    summary = perf_report.summarize(tmp_path)
    assert summary["worker_seconds"]["total"] == 4.0
    assert summary["elapsed"]["session_seconds"] == 2.0
    assert {row["worker_seconds"] for row in summary["worker_seconds"]["by_worker"].values()} == {
        2.0
    }
    assert summary["files"] == [
        {"file": "a.py", "worker_seconds": 2.0, "tests": 2},
        {"file": "b.py", "worker_seconds": 2.0, "tests": 2},
    ]
    text = perf_report.render_text(summary)
    assert "elapsed (wall clock): session 2.0s" in text
    assert "worker-seconds (summed across 2 worker(s); not wall time): 4.0" in text


def test_a_killed_run_is_summarized_from_its_flushed_records(tmp_path):
    (tmp_path / "inventory-gw0-2.json").write_text(
        json.dumps(
            {
                "collected": ["a.py::t1", "a.py::t2", "a.py::t3"],
                "selected": ["a.py::t1", "a.py::t2", "a.py::t3"],
                "deselected": [],
                "counts": {"collected": 3, "selected": 3, "deselected": 0},
            }
        ),
        encoding="utf-8",
    )
    _write_events(
        tmp_path / "events-controller-1.jsonl",
        [
            {
                "kind": "session_start",
                "mono": 100.0,
                "worker": "controller",
                "role": "controller",
                "pid": 1,
            }
        ],
    )
    worker = _worker_records("gw0", 2, [("a.py::t1", 1.0)])
    worker.append({"kind": "start", "mono": 101.0, "nodeid": "a.py::t2"})
    # Killed mid-line: the truncated record is counted, not fatal.
    _write_events(tmp_path / "events-gw0-2.jsonl", worker, tail='{"v":1,"kind":"pha')

    summary = perf_report.summarize(tmp_path)
    assert summary["status"] == "unfinished"
    assert summary["malformed_lines"] == 1
    assert summary["tests"]["unfinished"] == [{"nodeid": "a.py::t2", "worker": "gw0"}]
    assert (
        summary["tests"]["started"],
        summary["tests"]["finished"],
        summary["tests"]["not_started"],
    ) == (2, 1, 1)
    assert summary["worker_seconds"]["by_worker"]["gw0"]["last_active"] == "a.py::t2"
    assert summary["elapsed"]["session_seconds"] is None
    assert summary["elapsed"]["observed_span_seconds"] == 1.0

    # The wrapper having been killed too (runner still "running") reads as an
    # external kill — a CI step timeout or cancellation — not a pytest verdict.
    (tmp_path / perf_report.RUNNER).write_text(json.dumps({"state": "running"}), encoding="utf-8")
    assert perf_report.summarize(tmp_path)["status"] == "killed_externally"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["git", "-C", "/secret/repo", "-c", "user.name=x", "--no-pager", "rev-parse", "HEAD"],
            "git rev-parse",
        ),
        (["C:\\Program Files\\Git\\cmd\\git.exe", "ls-files", "-z"], "git ls-files"),
        (["git", "--version"], "git <no-verb>"),
        (["git", "/tmp/not-a-verb"], "git <other>"),
        (["/usr/bin/python3.13", "-c", "print(1)"], "python"),
        ('"C:\\Program Files\\Python\\pythonw.exe" -c pass', "python"),
        ("tmux new-session -d", "tmux"),
        # win32 audits the joined command line, not the list.
        ('"C:\\Program Files\\Git\\cmd\\git.exe" -c "a=b c" --no-pager status', "git status"),
        ("git -c metrics.canary=x version", "git version"),
        ('git "/tmp/not a verb"', "git <other>"),
        ('"C:\\unterminated\\git.exe', "git <no-verb>"),
        (["/tmp/some dir/odd name!"], "<other>"),
    ],
)
def test_commands_are_reduced_to_program_and_git_verb(args, expected):
    assert perf_report.describe_command(None, args) == expected


# -------------------------------------------------------------- the wrapper


def _gone(pid: int) -> bool:
    """Dead, or a zombie waiting for its (new) parent to reap it."""
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError):
            return True
        return stat.rsplit(")", 1)[1].split()[0] in ("Z", "X")
    return not get_process_host().is_alive(pid)


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _gone(pid):
            return True
        time.sleep(0.05)
    return _gone(pid)


@pytest.fixture
def reap_leftovers() -> Iterator[list[int]]:
    """Pids a failing row may leak; force-killed at teardown so a red row never
    leaves a 120s sleeper behind."""
    pids: list[int] = []
    yield pids
    host = get_process_host()
    for pid in pids:
        if not _gone(pid):
            with contextlib.suppress(Exception):
                host.force_kill(pid)


SLEEPER_SUITE = """\
import os
import subprocess
import sys
import time
from pathlib import Path


def test_first():
    pass


def test_sleeper():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    Path(os.environ["METRICS_PIDS"]).write_text(f"{os.getpid()} {child.pid}", encoding="utf-8")
    time.sleep(120)
"""

WRAPPER_DEADLINE_S = 15.0
# POSIX also gets the graceful path (SIGINT, then SIGKILL after the grace). On
# Windows pytest shares the wrapper's console group, so no Ctrl+C can be aimed
# at it alone: the deadline kills the tree at once and the grace does not apply.
GRACE_ROWS = [0.0] if sys.platform == "win32" else [0.0, 10.0]


@pytest.mark.parametrize("grace", GRACE_ROWS)
def test_wrapper_deadline_stops_the_tree_and_keeps_partial_records(tmp_path, reap_leftovers, grace):
    suite = _suite(tmp_path / "suite", {"test_sleep.py": SLEEPER_SUITE})
    metrics = tmp_path / "metrics"
    pids_file = tmp_path / "pids"
    started = time.monotonic()
    proc = _wrapped(
        suite,
        metrics,
        "-q",
        wrapper_args=("--timeout", str(WRAPPER_DEADLINE_S), "--grace", str(grace)),
        METRICS_PIDS=str(pids_file),
    )
    elapsed = time.monotonic() - started
    assert pids_file.exists(), f"the sleeper never started before the deadline\n{_output(proc)}"
    pids = [int(token) for token in pids_file.read_text(encoding="utf-8").split()]
    reap_leftovers.extend(pids)

    assert proc.returncode == run_test_benchmark.EXIT_TIMED_OUT, _output(proc)
    # Bounded: the deadline, the grace, the wrapper's own reap wait, and startup.
    assert elapsed < WRAPPER_DEADLINE_S + grace + 30, elapsed
    for pid in pids:
        assert _wait_gone(pid), f"pid {pid} outlived the wrapper's tree kill"

    runner = _json(metrics / perf_report.RUNNER)
    assert runner["state"] == "timed_out"
    assert runner["child_reaped"] is True
    if sys.platform == "win32":
        assert runner["tree_job"] is True, "the tree ran outside its job object"
    summary = _json(metrics / perf_report.SUMMARY_JSON)
    assert summary["status"] == "timed_out"
    assert summary["tests"]["unfinished"] == [
        {"nodeid": "test_sleep.py::test_sleeper", "worker": "main"}
    ]
    assert summary["tests"]["outcomes"] == {"passed": 1, "unfinished": 1}
    finish = [
        r for records in _events(metrics).values() for r in records if r["kind"] == "session_finish"
    ]
    if grace:
        # SIGINT first: pytest still closed its session, as interrupted.
        assert [r["exitstatus"] for r in finish] == [2]
    else:
        assert finish == []


def test_wrapper_reports_a_spawn_failure(tmp_path, monkeypatch):
    def no_spawn(*_args, **_kwargs):
        raise FileNotFoundError("no interpreter")

    monkeypatch.setattr(run_test_benchmark.subprocess, "Popen", no_spawn)
    rc = run_test_benchmark.main(["--metrics-dir", str(tmp_path / "m"), "--", "-q"])
    assert rc == run_test_benchmark.EXIT_SPAWN_FAILED
    runner = _json(tmp_path / "m" / perf_report.RUNNER)
    assert (runner["state"], runner["error"], runner["exit_code"]) == (
        "spawn_failed",
        "FileNotFoundError",
        None,
    )
    assert _json(tmp_path / "m" / perf_report.SUMMARY_JSON)["status"] == "spawn_failed"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal-to-self and killpg path")
def test_wrapper_finishes_cleanup_through_a_second_cancel_signal(tmp_path, monkeypatch):
    """A CI cancel is SIGINT then SIGTERM, the second landing inside the grace
    wait. It must not abort the tree kill: the group still gets its SIGKILL and
    the record still closes as interrupted."""

    class CancelledPytest:
        pid = 424242
        returncode = None

        def __init__(self):
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:  # the run itself: the cancel's SIGINT
                os.kill(os.getpid(), signal.SIGINT)
            elif self.waits == 2:  # the grace wait: the cancel's SIGTERM
                os.kill(os.getpid(), signal.SIGTERM)
                raise subprocess.TimeoutExpired("pytest", timeout)
            else:
                self.returncode = -9
            return self.returncode

    killed = []
    monkeypatch.setattr(run_test_benchmark.subprocess, "Popen", lambda *a, **k: CancelledPytest())
    monkeypatch.setattr(run_test_benchmark.os, "killpg", lambda pgid, sig: killed.append(sig))
    rc = run_test_benchmark.main(["--metrics-dir", str(tmp_path / "m"), "--grace", "5", "--", "-q"])

    assert rc == run_test_benchmark.EXIT_INTERRUPTED
    assert killed == [signal.SIGINT, signal.SIGKILL]
    runner = _json(tmp_path / "m" / perf_report.RUNNER)
    assert (runner["state"], runner["child_reaped"]) == ("interrupted", True)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal-to-self and killpg path")
def test_wrapper_stops_the_tree_on_a_signal_that_lands_during_the_spawn(tmp_path, monkeypatch):
    """A cancel landing while Popen is still returning — the child already
    exists in its own session, so the signal never reached it — must still
    reach the tree kill, not escape past it and leave pytest running with the
    record at ``running``.

    Ablation: install the raising handler before the spawn and this row
    escapes ``main`` with nothing killed."""

    class SpawnedMidSignal:
        pid = 424243
        returncode = None

        def __init__(self, *_args, **_kwargs):
            os.kill(os.getpid(), signal.SIGTERM)  # the child is born; the cancel lands

        def wait(self, timeout=None):
            if timeout == 5.0:  # the grace wait after the group SIGINT
                raise subprocess.TimeoutExpired("pytest", timeout)
            self.returncode = -9
            return self.returncode

    killed = []
    monkeypatch.setattr(run_test_benchmark.subprocess, "Popen", SpawnedMidSignal)
    monkeypatch.setattr(run_test_benchmark.os, "killpg", lambda pgid, sig: killed.append(sig))
    rc = run_test_benchmark.main(["--metrics-dir", str(tmp_path / "m"), "--grace", "5", "--", "-q"])

    assert rc == run_test_benchmark.EXIT_INTERRUPTED
    assert killed == [signal.SIGINT, signal.SIGKILL]
    runner = _json(tmp_path / "m" / perf_report.RUNNER)
    assert (runner["state"], runner["child_pid"]) == ("interrupted", 424243)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal-to-self and killpg path")
def test_wrapper_stops_the_tree_on_a_signal_that_lands_as_the_handlers_go_in(tmp_path, monkeypatch):
    """A cancel landing just after the first raising handler is installed —
    before the wait is reached — must still reach the tree kill, not escape to
    the outer ``finally``, which only restores handlers.

    Ablation: install the raising handlers ahead of the guarded ``try`` and
    ``_Interrupted`` escapes ``main`` with nothing killed."""

    class Spawned:
        pid = 424247
        returncode = None

        def wait(self, timeout=None):
            if timeout == 5.0:  # the grace wait after the group SIGINT
                raise subprocess.TimeoutExpired("pytest", timeout)
            self.returncode = -9
            return self.returncode

    real_signal = signal.signal
    fired = []

    def install(sig, handler):
        previous = real_signal(sig, handler)
        if handler is run_test_benchmark._raise_interrupted and not fired:
            fired.append(sig)
            os.kill(os.getpid(), sig)  # lands before the SIGTERM handler goes in
        return previous

    killed = []
    monkeypatch.setattr(run_test_benchmark.subprocess, "Popen", lambda *a, **k: Spawned())
    monkeypatch.setattr(run_test_benchmark.signal, "signal", install)
    monkeypatch.setattr(run_test_benchmark.os, "killpg", lambda pgid, sig: killed.append(sig))
    rc = run_test_benchmark.main(["--metrics-dir", str(tmp_path / "m"), "--grace", "5", "--", "-q"])

    assert fired == [signal.SIGINT]
    assert rc == run_test_benchmark.EXIT_INTERRUPTED
    assert killed == [signal.SIGINT, signal.SIGKILL]
    runner = _json(tmp_path / "m" / perf_report.RUNNER)
    assert (runner["state"], runner["child_pid"]) == ("interrupted", 424247)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX killpg path")
def test_wrapper_stops_the_tree_when_post_spawn_bookkeeping_raises(tmp_path, monkeypatch):
    """A fault after the spawn that is not a signal — the ``child_pid`` record
    write failing on a full metrics volume — must still stop the tree before it
    propagates: pytest runs in a session of its own and would outlive the
    wrapper.

    Ablation: drop the catch-all handler after the spawn and nothing is killed."""

    class Spawned:
        pid = 424244
        returncode = None

        def wait(self, timeout=None):
            if timeout == 5.0:  # the grace wait after the group SIGINT
                raise subprocess.TimeoutExpired("pytest", timeout)
            self.returncode = -9
            return self.returncode

    real_write = run_test_benchmark._write_runner

    def write_runner(root, record):
        if "child_pid" in record:
            raise OSError(28, "No space left on device")
        real_write(root, record)

    killed = []
    monkeypatch.setattr(run_test_benchmark.subprocess, "Popen", lambda *a, **k: Spawned())
    monkeypatch.setattr(run_test_benchmark, "_write_runner", write_runner)
    monkeypatch.setattr(run_test_benchmark.os, "killpg", lambda pgid, sig: killed.append(sig))
    with pytest.raises(OSError, match="No space left"):
        run_test_benchmark.main(["--metrics-dir", str(tmp_path / "m"), "--grace", "5", "--", "-q"])

    assert killed == [signal.SIGINT, signal.SIGKILL]
    assert _json(tmp_path / "m" / perf_report.RUNNER)["state"] == "running"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal-to-self")
def test_wrapper_writes_the_exit_record_through_a_late_cancel(tmp_path, monkeypatch):
    """A cancel landing after pytest exited, while the wrapper writes its
    terminal record, must not abort that write: the record closes as
    ``exited`` and the caller's handler is back once ``main`` returns.

    Ablation: write the record after the handlers are restored and the caller's
    handler raises out of ``main`` with the record at ``running``."""

    class LateCancel(Exception):
        pass

    def caller_handler(signum, frame):
        raise LateCancel(signum)

    class Spawned:
        pid = 424246
        returncode = 0

        def wait(self, timeout=None):
            return self.returncode

    real_write = run_test_benchmark._write_runner

    def write_runner(root, record):
        if record["state"] == "exited":
            os.kill(os.getpid(), signal.SIGTERM)
        real_write(root, record)

    monkeypatch.setattr(run_test_benchmark.subprocess, "Popen", lambda *a, **k: Spawned())
    monkeypatch.setattr(run_test_benchmark, "_write_runner", write_runner)
    previous = signal.signal(signal.SIGTERM, caller_handler)
    try:
        rc = run_test_benchmark.main(["--metrics-dir", str(tmp_path / "m"), "--", "-q"])
        assert signal.getsignal(signal.SIGTERM) is caller_handler
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert rc == 0
    runner = _json(tmp_path / "m" / perf_report.RUNNER)
    assert (runner["state"], runner["exit_code"]) == ("exited", 0)


def test_wrapper_wait_counts_setup_time_against_the_deadline(tmp_path, monkeypatch):
    """The deadline is one fixed instant from the wrapper's start: time spent
    before the wait (here a slow ``child_pid`` record write) comes out of the
    budget, so the stop cannot land past ``--deadline-epoch``.

    Ablation: pass the initial budget to ``proc.wait`` and it gets 100, not 60."""
    clock = [1000.0]
    waits = []

    class Spawned:
        pid = 424245
        returncode = 0

        def wait(self, timeout=None):
            waits.append(timeout)
            return self.returncode

    real_write = run_test_benchmark._write_runner

    def slow_write(root, record):
        if "child_pid" in record and record["state"] == "running":
            clock[0] += 40.0
        real_write(root, record)

    fake_time = type("FakeTime", (), {"time": staticmethod(time.time)})()
    fake_time.monotonic = lambda: clock[0]
    monkeypatch.setattr(run_test_benchmark, "time", fake_time)
    monkeypatch.setattr(run_test_benchmark, "_WAIT_POLL_S", None)  # one wait, as on POSIX
    monkeypatch.setattr(run_test_benchmark.subprocess, "Popen", lambda *a, **k: Spawned())
    monkeypatch.setattr(run_test_benchmark, "_write_runner", slow_write)
    rc = run_test_benchmark.main(
        ["--metrics-dir", str(tmp_path / "m"), "--timeout", "100", "--", "-q"]
    )

    assert rc == 0
    assert waits == [60.0]


def test_wrapper_sliced_wait_stops_at_the_deadline_not_past_it(tmp_path, monkeypatch):
    """Windows waits in slices; the last slice is cut to what is left of the
    budget, and only the wait that reaches the deadline stops the tree.

    Ablation: re-raise every slice's ``TimeoutExpired`` and the run times out
    after the first 25 seconds."""
    clock = [1000.0]
    waits = []

    class Stuck:
        pid = 424248
        returncode = None

        def wait(self, timeout=None):
            if self.returncode is not None or timeout == run_test_benchmark._REAP_WAIT_S:
                self.returncode = -9  # the tree kill's reap
                return self.returncode
            waits.append(timeout)
            clock[0] += timeout
            raise subprocess.TimeoutExpired("pytest", timeout)

    fake_time = type("FakeTime", (), {"time": staticmethod(time.time)})()
    fake_time.monotonic = lambda: clock[0]
    monkeypatch.setattr(run_test_benchmark, "time", fake_time)
    monkeypatch.setattr(run_test_benchmark, "_WAIT_POLL_S", 25.0)
    monkeypatch.setattr(run_test_benchmark.subprocess, "Popen", lambda *a, **k: Stuck())
    monkeypatch.setattr(
        run_test_benchmark,
        "_kill_tree",
        lambda proc, grace, job=None: proc.wait(run_test_benchmark._REAP_WAIT_S),
    )
    rc = run_test_benchmark.main(
        ["--metrics-dir", str(tmp_path / "m"), "--timeout", "60", "--", "-q"]
    )

    assert rc == run_test_benchmark.EXIT_TIMED_OUT
    assert waits == [25.0, 25.0, 10.0]
    assert _json(tmp_path / "m" / perf_report.RUNNER)["state"] == "timed_out"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal-to-self and killpg path")
def test_wrapper_reaches_the_tree_kill_while_pytest_hangs_on_the_cancel(tmp_path, monkeypatch):
    """Windows cannot interrupt ``Popen.wait``: a Ctrl+C runs the wrapper's
    handler only when a wait returns. With no deadline, a pytest stuck on the
    cancel must still be stopped between slices, never waited on unbounded.

    Ablation: wait once with ``timeout=None`` and the unbounded wait is refused
    here (the real one would block for good)."""

    class HungOnCancel:
        pid = 424249
        returncode = None

        def __init__(self):
            self.slices = 0

        def wait(self, timeout=None):
            assert timeout is not None, "unbounded wait: a Windows Ctrl+C is never handled"
            if self.returncode is not None or timeout == run_test_benchmark._REAP_WAIT_S:
                self.returncode = -9
                return self.returncode
            self.slices += 1
            if self.slices == 3:  # the cancel: its handler runs as this slice ends
                os.kill(os.getpid(), signal.SIGINT)
            raise subprocess.TimeoutExpired("pytest", timeout)

    killed = []
    monkeypatch.setattr(run_test_benchmark, "_WAIT_POLL_S", 0.5)
    monkeypatch.setattr(run_test_benchmark.subprocess, "Popen", lambda *a, **k: HungOnCancel())
    monkeypatch.setattr(run_test_benchmark.os, "killpg", lambda pgid, sig: killed.append(sig))
    rc = run_test_benchmark.main(["--metrics-dir", str(tmp_path / "m"), "--grace", "0", "--", "-q"])

    assert rc == run_test_benchmark.EXIT_INTERRUPTED
    assert killed == [signal.SIGKILL]
    runner = _json(tmp_path / "m" / perf_report.RUNNER)
    assert (runner["state"], runner["child_reaped"]) == ("interrupted", True)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job-object path")
def test_wrapper_tree_kill_reaches_a_ctrl_c_immune_child_after_the_root_exited(
    tmp_path, reap_leftovers
):
    """An interrupt reaches the wrapper only once pytest has exited
    (``Popen.wait`` is not interruptible on Windows), when ``taskkill /T`` can no
    longer walk from the reaped root. A child in its own process group ignored
    the Ctrl+C; the job must still reach it. The root is started through
    ``sys.executable`` — on CI the venv redirector, whose own job hands every
    grandchild a silent breakaway — and the job lives in a driver process so this
    worker never joins one."""
    pid_file = tmp_path / "child"
    root = (
        "import pathlib, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],"
        " creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid), encoding='utf-8')\n"
    )
    driver = (
        "import subprocess, sys\n"
        f"sys.path.insert(0, {str(WRAPPER.parent)!r})\n"
        "import run_test_benchmark as rtb\n"
        "job = rtb._TreeJob.enter()\n"
        f"proc = subprocess.Popen([sys.executable, '-c', {root!r}])\n"
        "assert proc.wait(timeout=60) == 0\n"
        "print(open(sys.argv[1], encoding='utf-8').read(), flush=True)\n"
        "input()  # the test checks the child is alive, then releases the kill\n"
        "rtb._kill_tree(proc, 0.0, job)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", driver, str(pid_file)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        line = proc.stdout.readline()
        assert line.strip().isdigit(), f"driver failed before the kill: {line!r}"
        child = int(line)
        reap_leftovers.append(child)
        assert not _gone(child), "the child must outlive its root for this row to mean anything"
        proc.stdin.write("\n")
        proc.stdin.flush()
        assert proc.wait(timeout=CHILD_TIMEOUT_S) == 0
        assert _wait_gone(child), f"pid {child} outlived the tree kill"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def test_wrapper_refuses_a_used_directory_and_a_passed_deadline(tmp_path, monkeypatch):
    def no_spawn(*_args, **_kwargs):
        raise AssertionError("must not start pytest")

    monkeypatch.setattr(run_test_benchmark.subprocess, "Popen", no_spawn)
    used = tmp_path / "used"
    used.mkdir()
    (used / perf_report.RUNNER).write_text("{}", encoding="utf-8")
    assert (
        run_test_benchmark.main(["--metrics-dir", str(used), "--", "-q"])
        == run_test_benchmark.EXIT_USAGE
    )

    late = tmp_path / "late"
    rc = run_test_benchmark.main(
        ["--metrics-dir", str(late), "--deadline-epoch", str(time.time() - 1), "--", "-q"]
    )
    assert rc == run_test_benchmark.EXIT_TIMED_OUT
    runner = _json(late / perf_report.RUNNER)
    assert (runner["state"], runner["child_started"]) == ("timed_out", False)


# ------------------------------------------------------------------- CI shards

# Two files, so a function family and an xdist group each have a parent nodeid to
# straddle: `test_b.py` repeats a function name from `test_a.py` (a different
# family), and both files carry members of one group.
SHARD_SUITE_A = """\
import pytest


@pytest.mark.parametrize("n", range(7))
def test_params(n):
    pass


class TestMethods:
    @pytest.mark.parametrize("m", ["x", "y", "z"])
    def test_method(self, m):
        pass

    def test_plain(self):
        pass


@pytest.mark.xdist_group("real-mux")
@pytest.mark.parametrize("k", range(3))
def test_grouped_a(k):
    pass


@pytest.mark.skip(reason="stands in for a platform skip: selected, reported, not run")
def test_platform_skip():
    pass
""" + "".join(f"\n\ndef test_fn_{i}():\n    pass\n" for i in range(24))

SHARD_SUITE_B = """\
import pytest


@pytest.mark.parametrize("n", range(4))
def test_params(n):
    pass


@pytest.mark.xdist_group("real-mux")
def test_grouped_b():
    pass
""" + "".join(f"\n\ndef test_other_{i}():\n    pass\n" for i in range(24))


def _collected(suite: Path, *args: str, **env: str) -> list[str]:
    proc = _pytest(suite, "-q", "--collect-only", *args, **env)
    assert proc.returncode == 0, _output(proc)
    return [line for line in proc.stdout.splitlines() if "::" in line and " " not in line]


@pytest.fixture(scope="module")
def shard_suite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _suite(
        tmp_path_factory.mktemp("shards") / "suite",
        {"test_a.py": SHARD_SUITE_A, "test_b.py": SHARD_SUITE_B},
    )


@pytest.mark.parametrize(
    "spec", ["0/2", "3/2", "1/0", "0/0", "a/b", "1", "1/2/3", "-1/2", " / ", ""]
)
def test_ci_shard_rejects_malformed_ranges(spec):
    with pytest.raises(pytest.UsageError, match="--ci-shard expects I/N"):
        perf_report.parse_shard(spec)


def test_ci_shard_accepts_its_bounds():
    assert perf_report.parse_shard("1/1") == (1, 1)
    assert perf_report.parse_shard(" 2/2 ") == (2, 2)
    assert perf_report.parse_shard("7/10") == (7, 10)


def test_a_malformed_shard_is_a_usage_error_before_any_test_runs(shard_suite):
    proc = _pytest(shard_suite, "-q", "--ci-shard=3/2")
    assert proc.returncode == 4, _output(proc)
    assert "--ci-shard expects I/N" in proc.stderr
    assert " passed" not in proc.stdout


def test_ci_shards_partition_the_suite_by_function_family_and_group(shard_suite):
    full = _collected(shard_suite)
    assert len(full) == 7 + 3 + 1 + 3 + 1 + 24 + 4 + 1 + 24
    shards = [set(_collected(shard_suite, f"--ci-shard={i}/3")) for i in (1, 2, 3)]
    assert all(shards), "every shard of this suite is non-empty"
    assert sum(len(s) for s in shards) == len(full)
    assert set().union(*shards) == set(full)

    def owners(predicate) -> set[int]:
        return {i for i, s in enumerate(shards) for nodeid in s if predicate(nodeid)}

    # Parameter variants never split; same-named functions in two files are two families.
    for fam in {perf_report.family(nodeid) for nodeid in full}:
        assert len(owners(lambda n, fam=fam: perf_report.family(n) == fam)) == 1, fam
    # One xdist group is one unit, across files and parametrizations.
    assert len(owners(lambda n: "test_grouped_" in n)) == 1
    # The skip is selected like any other test (it is reported, not dropped).
    assert len(owners(lambda n: n.endswith("test_platform_skip"))) == 1


def test_ci_shard_selection_ignores_hash_randomization(shard_suite):
    first = _collected(shard_suite, "--ci-shard=1/2", PYTHONHASHSEED="1")
    again = _collected(shard_suite, "--ci-shard=1/2", PYTHONHASHSEED="2")
    assert first == again
    unit = "function:test_a.py::test_params"
    assert (
        perf_report.shard_of(unit, 2)
        == int(__import__("hashlib").sha256(unit.encode()).hexdigest()[:16], 16) % 2
    )


def test_a_new_test_is_assigned_without_moving_existing_ones(tmp_path, shard_suite):
    before = [set(_collected(shard_suite, f"--ci-shard={i}/2")) for i in (1, 2)]
    grown = _suite(
        tmp_path / "grown",
        {
            "test_a.py": SHARD_SUITE_A,
            "test_b.py": SHARD_SUITE_B,
            "test_new.py": "def test_added():\n    pass\n",
        },
    )
    after = [set(_collected(grown, f"--ci-shard={i}/2")) for i in (1, 2)]
    added = "test_new.py::test_added"
    assert [added in shard for shard in after].count(True) == 1
    assert [shard - {added} for shard in after] == before


def test_an_empty_shard_fails_loudly(tmp_path):
    suite = _suite(tmp_path / "suite", {"test_one.py": "def test_one():\n    pass\n"})
    home = perf_report.shard_of("function:test_one.py::test_one", 2) + 1
    ok = _pytest(suite, "-q", f"--ci-shard={home}/2")
    assert ok.returncode == 0, _output(ok)
    empty = _pytest(suite, "-q", f"--ci-shard={3 - home}/2")
    assert empty.returncode != 0, _output(empty)
    assert "selected none of 1 tests" in empty.stdout + empty.stderr


def _shard_run(suite: Path, metrics: Path, spec: str, *args: str) -> Path:
    proc = _wrapped(suite, metrics, "-q", f"--ci-shard={spec}", *args)
    assert proc.returncode in (0, 1), _output(proc)
    return metrics


@pytest.fixture(scope="module")
def shard_runs(tmp_path_factory, shard_suite) -> tuple[Path, Path]:
    """Both halves of a real 2-shard run; the second under xdist, as in CI."""
    base = tmp_path_factory.mktemp("shard-runs")
    one = _shard_run(shard_suite, base / "one", "1/2")
    two = _shard_run(shard_suite, base / "two", "2/2", "-n", "2", "--dist", "loadgroup")
    return one, two


def test_verify_shards_accepts_a_complete_green_partition(shard_runs):
    assert perf_report.verify_shards(list(shard_runs), 2) == []
    assert perf_report.main(["verify-shards", "--count", "2", *map(str, shard_runs)]) == 0


def test_verify_shards_fails_a_missing_shard(shard_runs, tmp_path):
    one, _ = shard_runs
    assert perf_report.verify_shards([one], 2) == ["shard 2/2: no results"]
    # An artifact that never arrived: the directory exists but holds nothing.
    absent = tmp_path / "absent"
    absent.mkdir()
    problems = perf_report.verify_shards([one, absent], 2)
    assert any("no manifest" in p for p in problems), problems
    assert "shard 2/2: no results" in problems
    assert perf_report.main(["verify-shards", "--count", "2", str(one)]) == 1


def test_verify_shards_fails_an_omitted_test(shard_suite, shard_runs, tmp_path):
    # A shard that selected one test less than its share: collected still lists
    # it, so only the union-versus-collected parity check can notice.
    # Ablation: dropping the `omitted` check in verify_shards makes this fail.
    one, two = shard_runs
    victim = sorted(_json_inventory(two)["selected"])[0]
    short = _shard_run(shard_suite, tmp_path / "short", "2/2", "--deselect", victim)
    problems = perf_report.verify_shards([one, short], 2)
    assert problems == [f"1 collected tests ran in no shard: {victim}"]


def test_verify_shards_fails_a_failed_or_cancelled_shard(tmp_path, shard_runs):
    one, _ = shard_runs
    suite = _suite(
        tmp_path / "red",
        {
            "test_a.py": SHARD_SUITE_A.replace(
                "def test_fn_0():\n    pass", "def test_fn_0():\n    assert False"
            ),
            "test_b.py": SHARD_SUITE_B,
        },
    )
    home = perf_report.shard_of("function:test_a.py::test_fn_0", 2) + 1
    red = _shard_run(suite, tmp_path / "red-metrics", f"{home}/2")
    problems = perf_report.verify_shards([red], 2)
    assert any("tests_failed" in p for p in problems), problems

    # Cancelled: the runner never recorded an exit and pytest no session_finish.
    cut = tmp_path / "cut"
    shutil.copytree(one, cut)
    for events in cut.glob("events-*.jsonl"):
        kept = [
            line
            for line in events.read_text(encoding="utf-8").splitlines()
            if '"kind":"session_finish"' not in line
        ]
        events.write_text("\n".join(kept) + "\n", encoding="utf-8")
    runner = _json(cut / perf_report.RUNNER)
    runner.update(state="running", exit_code=None)
    (cut / perf_report.RUNNER).write_text(json.dumps(runner), encoding="utf-8")
    problems = perf_report.verify_shards([cut], 2)
    assert any("killed_externally" in p for p in problems), problems

    # A green session_finish is not a green process: the wrapper's record of what
    # happened after it (killed from outside; a failed or killed cleanup) wins.
    for label, state, exit_code, expected in (
        ("late-kill", "running", None, "killed_externally"),
        ("late-crash", "exited", 1, "exit_mismatch"),
        ("late-signal", "exited", -9, "exit_mismatch"),
    ):
        late = tmp_path / label
        shutil.copytree(one, late)
        runner = _json(late / perf_report.RUNNER)
        runner.update(state=state, exit_code=exit_code)
        (late / perf_report.RUNNER).write_text(json.dumps(runner), encoding="utf-8")
        problems = perf_report.verify_shards([late], 2)
        assert any(expected in p for p in problems), (label, problems)

    # No settled wrapper record: a green session_finish alone proves nothing.
    # Ablation: drop the runner check in _run_problems and these rows pass green.
    for label, content, expected in (
        ("no-runner", None, "no readable runner.json"),
        ("torn-runner", '{"state": "exi', "no readable runner.json"),
        ("odd-runner", {"state": "exiting", "exit_code": 0}, "state 'exiting', not 'exited'"),
    ):
        bare = tmp_path / label
        shutil.copytree(one, bare)
        path = bare / perf_report.RUNNER
        if content is None:
            path.unlink()
        else:
            text = content if isinstance(content, str) else json.dumps(content)
            path.write_text(text, encoding="utf-8")
        problems = perf_report.verify_shards([bare], 2)
        assert any(expected in p for p in problems), (label, problems)


def test_verify_shards_fails_overlap_duplicates_and_foreign_runs(shard_runs, tmp_path):
    one, two = shard_runs
    assert "shard 1/2 reported twice" in " ".join(perf_report.verify_shards([one, one, two], 2))
    assert any("is not one of 3" in p for p in perf_report.verify_shards([one, two], 3))

    overlap = tmp_path / "overlap"
    shutil.copytree(two, overlap)
    path, inventory = _inventory_file(overlap)
    stolen = sorted(_json_inventory(one)["selected"])[0]
    inventory["selected"].append(stolen)
    path.write_text(json.dumps(inventory), encoding="utf-8")
    problems = perf_report.verify_shards([one, overlap], 2)
    assert f"1 tests ran in more than one shard: {stolen}" in problems
    assert any("straddle shards" in p for p in problems), problems


def test_verify_executed_refuses_a_gate_that_skipped(tmp_path):
    suite = _suite(
        tmp_path / "gate",
        {
            "test_live.py": "import pytest\n\n\ndef test_live():\n    pass\n\n\n"
            "@pytest.mark.skip(reason='binary absent')\ndef test_absent():\n    pass\n"
        },
    )
    skipped = _wrapped(suite, tmp_path / "skipped", "-q")
    assert skipped.returncode == 0, _output(skipped)
    problems = perf_report.verify_executed(tmp_path / "skipped")
    assert problems == [
        f"{tmp_path / 'skipped'}: not every selected test passed: {{'skipped': 1}}",
        f"{tmp_path / 'skipped'}: 1 of 2 tests passed",
    ]
    ran = _wrapped(suite, tmp_path / "ran", "-q", "-k", "not absent")
    assert ran.returncode == 0, _output(ran)
    assert perf_report.verify_executed(tmp_path / "ran") == []
    assert perf_report.main(["verify-executed", str(tmp_path / "skipped")]) == 1


def test_verify_jobs_makes_the_latest_job_execution_authoritative(tmp_path):
    shard, gate = "test (windows, py3.11, shard 1/2)", "psmux gate (windows, py3.11)"

    def listing(*jobs: tuple[str, int, str | None]) -> Path:
        path = tmp_path / f"jobs-{uuid.uuid4().hex}.json"
        page = [
            {"name": n, "run_attempt": a, "status": "completed", "conclusion": c}
            for n, a, c in jobs
        ]
        # `--slurp` output: one array of pages; split across two pages.
        path.write_text(json.dumps([{"jobs": page[:1]}, {"jobs": page[1:]}]), encoding="utf-8")
        return path

    green = listing((shard, 1, "success"), (gate, 1, "success"))
    assert perf_report.verify_jobs(green, [shard, gate]) == []
    assert perf_report.main(["verify-jobs", str(green), "--expect", shard, "--expect", gate]) == 0
    # A re-run whose shard failed before uploading: attempt 1's green artifact
    # would still be downloadable, so the attempt-2 failure has to decide.
    rerun = listing((shard, 1, "success"), (shard, 2, "failure"), (gate, 2, "success"))
    assert perf_report.verify_jobs(rerun, [shard, gate]) == [
        f"job {shard!r}: attempt 2 is completed/failure, not a success -- "
        "its downloaded records may be an earlier attempt's"
    ]
    # Carried over by "re-run failed jobs": an earlier attempt's success stands.
    carried = listing((gate, 2, "success"), (shard, 1, "success"))
    assert perf_report.verify_jobs(carried, [shard, gate]) == []
    for conclusion in ("cancelled", "timed_out", "skipped", None):
        assert perf_report.verify_jobs(
            listing((shard, 1, conclusion), (gate, 1, "success")), [shard, gate]
        )
    assert perf_report.verify_jobs(
        listing((gate, 1, "success"), (gate, 1, "success")), [shard, gate]
    ) == [f"job {shard!r}: not in the run's latest executions"]
    torn = tmp_path / "torn.json"
    torn.write_text('[{"jobs": [', encoding="utf-8")
    assert perf_report.verify_jobs(torn, [shard])[0].startswith(f"{torn}: unreadable job listing")
    assert perf_report.verify_jobs(tmp_path / "absent.json", [shard])[0].startswith(
        f"{tmp_path / 'absent.json'}: unreadable job listing"
    )


def _inventory_file(root: Path) -> tuple[Path, dict[str, Any]]:
    for path in sorted(root.glob(f"{perf_report.INVENTORY_PREFIX}*.json")):
        data = _json(path)
        if "selected" in data:
            return path, data
    raise AssertionError(f"no full inventory in {root}")


def _json_inventory(root: Path) -> dict[str, Any]:
    return _inventory_file(root)[1]
