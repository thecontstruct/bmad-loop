#!/usr/bin/env python3
"""Run pytest with ``--test-metrics-dir`` and record the pytest process lifetime.

    uv run python scripts/run_test_benchmark.py --metrics-dir DIR [--timeout S |
        --deadline-epoch T] [--grace S] -- <pytest arguments>

The in-process metrics (tests/perf_report.py) cannot see the start of the
interpreter, or anything after its own ``atexit`` record: import time, the tail of
temp-directory cleanup, and a process that never exits at all. This wrapper owns
that span. It writes ``runner.json`` into the metrics directory before starting
pytest (``state: running``) and rewrites it when pytest ends, then regenerates
``summary.json``/``summary.txt`` and prints the text summary.

``runner.json`` states: ``running`` (the wrapper itself was killed, e.g. by a CI
step timeout or job cancellation), ``exited``, ``timed_out`` (the wrapper's own
deadline expired and it killed the pytest process tree), ``interrupted`` (the
wrapper got SIGINT/SIGTERM and stopped the tree), ``spawn_failed``.

Exit status: pytest's own exit status whenever pytest exits by itself (a POSIX
signal death maps to 128+signal), 124 when the deadline killed it, 130 when the
wrapper was interrupted, 127 when pytest could not be started, 2 for a wrapper
usage error. The wrapper never retries anything.

Deadline: ``--timeout`` is relative to the wrapper's start; ``--deadline-epoch``
is an absolute Unix time, which lets CI reserve the end of a job's
``timeout-minutes`` for uploading the metrics (see .github/workflows/ci.yml). On
expiry the tree is stopped: POSIX sends SIGINT to pytest's process group (pytest
then still writes its JUnit file and session records), waits ``--grace`` seconds
and SIGKILLs the group; Windows terminates the tree at once (pytest shares the
wrapper's console group there, as it would without the wrapper, so no Ctrl+C can
be aimed at it alone). Records already flushed by the metrics plugin survive
either way.

Windows tree: the wrapper enters a job object before starting pytest (see
``_TreeJob``), the stand-in for the POSIX process group — membership outlives
the root, so a child that ignored Ctrl+C (``CREATE_NEW_PROCESS_GROUP``) is still
reached after pytest itself exited and was reaped, when ``taskkill /T`` can no
longer walk to it. That is the normal case on an interrupt: a Ctrl+C reaches
pytest too, and usually ends it before the wrapper's own handler runs.
``Popen.wait`` is not interruptible there, so the wrapper waits in short slices
(``_WAIT_POLL_S``); a pytest stuck on the Ctrl+C is then still stopped, instead
of being waited on until the deadline, or forever without one. A stop runs
``taskkill /T /F`` first, then terminates every other job
member. If the job cannot be set up the run goes on without it and
``runner.json`` records ``tree_job: false``.

Standalone by design: stdlib only, and it does not import bmad_loop. Child output
passes straight through to this process's stdout/stderr and is never recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
RUNNER_FILE = "runner.json"
MANIFEST_FILE = "manifest.json"
TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"

EXIT_TIMED_OUT = 124
EXIT_SPAWN_FAILED = 127
EXIT_INTERRUPTED = 130
EXIT_USAGE = 2

# How long to wait for the direct child to be reaped after a tree kill.
_REAP_WAIT_S = 15.0
_TASKKILL_TIMEOUT_S = 30.0
# Windows: the longest a signal waits for the handler that raises it. The wait
# there blocks in WaitForSingleObject, which Python cannot interrupt; POSIX
# signals interrupt the wait itself, so it is one call.
_WAIT_POLL_S: float | None = 0.5 if sys.platform == "win32" else None


class _Interrupted(Exception):
    """SIGINT/SIGTERM reached the wrapper while pytest was running."""


class _TreeJob:
    """Windows only: a job object the wrapper enters before starting pytest, so
    pytest is born in it along with everything it starts — no window between
    the spawn and an assignment.

    No limits are set, so the job permits no breakaway. That is load-bearing:
    a venv's ``python.exe`` redirector runs the real interpreter in a job of its
    own with ``SILENT_BREAKAWAY_OK``, so every process pytest starts silently
    leaves that job and moves up the job chain until a job that does not permit
    breakaway — this one. (The flip side: a descendant that explicitly passes
    ``CREATE_BREAKAWAY_FROM_JOB`` is refused. Nothing under test does.) There is
    no kill-on-close either, so a run that exits normally leaves its processes
    as it would unwrapped. The wrapper stays a member, so the kill terminates
    every OTHER member, one handle at a time, never the job as a whole."""

    _PROCESS_TERMINATE = 0x0001
    # The full query right, not the limited one: Windows accepts either for
    # IsProcessInJob, Wine only this one, and members are our own processes.
    _PROCESS_QUERY_INFORMATION = 0x0400
    _JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
    _ERROR_MORE_DATA = 234
    _STILL_ACTIVE = 259
    _SWEEPS = 5  # members started while a sweep runs are caught by the next

    def __init__(self, kernel32: Any, handle: int) -> None:
        self._kernel32 = kernel32
        self._handle: int | None = handle

    @classmethod
    def enter(cls) -> _TreeJob:
        """Create the job and put the calling process in it; ``OSError`` on any
        failure (the process is then in no new job)."""
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = ()
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = (
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.IsProcessInJob.argtypes = (
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        )
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise cls._error("CreateJobObjectW")
        job = cls(kernel32, handle)
        if not kernel32.AssignProcessToJobObject(handle, kernel32.GetCurrentProcess()):
            error = cls._error("AssignProcessToJobObject")
            job.close()
            raise error
        return job

    @staticmethod
    def _error(what: str) -> OSError:
        import ctypes

        err = ctypes.get_last_error()
        return OSError(err, f"{what} failed (winerror {err})")

    def _member_pids(self) -> list[int]:
        import ctypes

        capacity = 256
        while True:
            # JOBOBJECT_BASIC_PROCESS_ID_LIST: two DWORD counts, then ULONG_PTR ids.
            class _List(ctypes.Structure):
                _fields_ = [
                    ("NumberOfAssignedProcesses", ctypes.c_uint32),
                    ("NumberOfProcessIdsInList", ctypes.c_uint32),
                    ("ProcessIdList", ctypes.c_size_t * capacity),
                ]

            ids = _List()
            if self._kernel32.QueryInformationJobObject(
                self._handle,
                self._JOB_OBJECT_BASIC_PROCESS_ID_LIST,
                ctypes.byref(ids),
                ctypes.sizeof(ids),
                None,
            ):
                return [int(pid) for pid in ids.ProcessIdList[: ids.NumberOfProcessIdsInList]]
            if ctypes.get_last_error() != self._ERROR_MORE_DATA:
                raise self._error("QueryInformationJobObject")
            capacity = max(capacity * 2, int(ids.NumberOfAssignedProcesses) + 64)

    def _terminate_member(self, pid: int) -> str | None:
        """Terminate ``pid`` if it is still a member; the failure, if any."""
        import ctypes
        from ctypes import wintypes

        access = self._PROCESS_TERMINATE | self._PROCESS_QUERY_INFORMATION
        process = self._kernel32.OpenProcess(access, False, pid)
        if not process:
            return None  # exited since the listing
        try:
            # The handle pins the pid: re-check membership so a pid reused
            # since the listing is never killed.
            member = wintypes.BOOL(False)
            if not self._kernel32.IsProcessInJob(process, self._handle, ctypes.byref(member)):
                return f"{pid}: IsProcessInJob winerror {ctypes.get_last_error()}"
            if not member or self._kernel32.TerminateProcess(process, 1):
                return None
            err = ctypes.get_last_error()
            # A member taskkill just ended is still listed until it finishes
            # exiting, and TerminateProcess refuses it (access denied): exited
            # is not a failure.
            code = wintypes.DWORD(self._STILL_ACTIVE)
            if (
                self._kernel32.GetExitCodeProcess(process, ctypes.byref(code))
                and code.value != self._STILL_ACTIVE
            ):
                return None
            return f"{pid}: TerminateProcess winerror {err}"
        finally:
            self._kernel32.CloseHandle(process)

    def terminate_others(self) -> None:
        """Terminate every member but the calling process. ``OSError`` if the
        listing fails or a member could not be checked or terminated — after
        every other member was still tried."""
        if self._handle is None:
            return
        me = os.getpid()
        failures: dict[int, str] = {}
        for _ in range(self._SWEEPS):
            others = [pid for pid in self._member_pids() if pid != me]
            if not others:
                break
            for pid in others:
                failure = self._terminate_member(pid)
                if failure is not None:
                    failures[pid] = failure
                else:
                    failures.pop(pid, None)
        if failures:
            raise OSError(f"job members not terminated: {'; '.join(failures.values())}")

    def close(self) -> None:
        if self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _write_runner(root: Path, record: dict[str, Any]) -> None:
    tmp = root / f".{RUNNER_FILE}.tmp"
    tmp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, root / RUNNER_FILE)


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pytest with --test-metrics-dir and record its process lifetime.",
        usage="%(prog)s --metrics-dir DIR [--timeout S | --deadline-epoch T] [--grace S] -- PYTEST_ARGS...",
    )
    parser.add_argument(
        "--metrics-dir", type=Path, required=True, help="fresh directory for all metrics files"
    )
    deadline = parser.add_mutually_exclusive_group()
    deadline.add_argument(
        "--timeout", type=float, help="seconds from now before the pytest tree is stopped"
    )
    deadline.add_argument(
        "--deadline-epoch", type=float, help="absolute Unix time at which the tree is stopped"
    )
    parser.add_argument(
        "--grace",
        type=float,
        default=20.0,
        help="POSIX: seconds between SIGINT and SIGKILL on expiry (default 20; 0 kills at once)",
    )
    parser.add_argument(
        "pytest_args", nargs=argparse.REMAINDER, help="arguments for pytest, after --"
    )
    args = parser.parse_args(argv)
    if args.pytest_args[:1] == ["--"]:
        args.pytest_args = args.pytest_args[1:]
    if args.grace < 0 or (args.timeout is not None and args.timeout <= 0):
        parser.error("--grace must be >= 0 and --timeout > 0")
    return args


def _kill_tree(proc: subprocess.Popen[bytes], grace: float, job: _TreeJob | None = None) -> None:
    """Stop pytest and everything it started. Bounded: returns within
    ``grace + _REAP_WAIT_S`` (+ taskkill's own bound on Windows)."""
    if sys.platform == "win32":
        # taskkill first, while a live root still anchors its parent-pid walk;
        # then the job, which still holds whatever that walk could not reach
        # (everything, once the root is reaped).
        try:
            subprocess.run(  # fixed argv, no shell
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_TASKKILL_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        if proc.poll() is None:
            proc.kill()
        if job is not None:
            try:
                job.terminate_others()
            except OSError as exc:
                sys.stderr.write(f"run_test_benchmark: job sweep failed ({exc})\n")
    else:
        pgid = proc.pid  # start_new_session=True made pytest its group leader
        if grace > 0:
            try:
                os.killpg(pgid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        # SIGKILL the group even if pytest itself exited: workers it left behind
        # still hold the group id.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.wait(timeout=_REAP_WAIT_S)
    except subprocess.TimeoutExpired:
        pass


def _wait_for_exit(proc: subprocess.Popen[bytes], deadline_mono: float | None) -> int:
    """``proc``'s exit status; ``subprocess.TimeoutExpired`` once ``deadline_mono``
    passes. Waits in ``_WAIT_POLL_S`` slices when set, so a pending signal's
    handler runs between them."""
    while True:
        remaining = None if deadline_mono is None else max(0.0, deadline_mono - time.monotonic())
        step = remaining
        if _WAIT_POLL_S is not None and (step is None or step > _WAIT_POLL_S):
            step = _WAIT_POLL_S
        try:
            return proc.wait(timeout=step)
        except subprocess.TimeoutExpired:
            if step == remaining:
                raise


def _raise_interrupted(signum: int, frame: object) -> None:
    raise _Interrupted(signum)


def _hold_signals_for_cleanup() -> None:
    """Stop a further SIGINT/SIGTERM from aborting the tree kill. A CI cancel
    sends SIGINT and then SIGTERM a few seconds later — inside the grace wait —
    and a second raise there would skip the group SIGKILL and leave the run
    record at ``running``. Cleanup is bounded, so nothing is lost by deferring:
    ``main``'s ``finally`` restores the caller's handlers."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)


def _summarize(root: Path) -> None:
    sys.path.insert(0, str(TESTS_DIR))
    try:
        import perf_report

        summary = perf_report.write_summary(root)
        sys.stdout.write("\n" + perf_report.render_text(summary))
        sys.stdout.flush()
    except Exception as exc:  # reporting must not mask pytest's exit status
        sys.stderr.write(
            f"run_test_benchmark: summary failed ({type(exc).__name__}: {exc}); "
            f"records in {root} are intact -- run `python tests/perf_report.py summarize {root}`\n"
        )
    finally:
        sys.path.remove(str(TESTS_DIR))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    root: Path = args.metrics_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / RUNNER_FILE).exists() or (root / MANIFEST_FILE).exists():
        sys.stderr.write(
            f"run_test_benchmark: {root} already holds a run; pass a fresh --metrics-dir\n"
        )
        return EXIT_USAGE

    start_wall = time.time()
    start_mono = time.monotonic()
    if args.deadline_epoch is not None:
        budget: float | None = args.deadline_epoch - start_wall
    else:
        budget = args.timeout
    # Both flags name one fixed instant; the wait below is measured against it,
    # so job setup, the spawn and the runner write come out of the budget
    # instead of pushing the stop past it into the upload reserve.
    deadline_mono = None if budget is None else start_mono + budget
    record: dict[str, Any] = {
        "v": SCHEMA_VERSION,
        "state": "running",
        "pid": os.getpid(),
        "python": platform.python_version(),
        "platform": sys.platform,
        "start_wall": start_wall,
        "start_mono": start_mono,
        "budget_seconds": budget,
        "grace_seconds": args.grace,
    }
    _write_runner(root, record)

    def finish(state: str, exit_code: int | None, **extra: Any) -> None:
        end_mono = time.monotonic()
        record.update(
            state=state,
            exit_code=exit_code,
            end_wall=time.time(),
            end_mono=end_mono,
            elapsed_seconds=round(end_mono - start_mono, 3),
            **extra,
        )
        _write_runner(root, record)

    if budget is not None and budget <= 0:
        sys.stderr.write(
            f"run_test_benchmark: deadline already passed ({budget:.0f}s); not starting pytest\n"
        )
        finish("timed_out", None, child_started=False)
        _summarize(root)
        return EXIT_TIMED_OUT

    command = [sys.executable, "-m", "pytest", f"--test-metrics-dir={root}", *args.pytest_args]
    # POSIX: a session of its own, so the whole tree can be signalled as a group.
    # Windows: deliberately NOT a new process group — that would disable Ctrl+C
    # for pytest and everything it starts, changing the environment under test
    # (an accidental CTRL_C_EVENT would stop aborting the run). `taskkill /T`
    # needs no group.
    popen_kwargs: dict[str, Any] = {} if sys.platform == "win32" else {"start_new_session": True}

    # Until Popen returns there is no child to stop, and a raise inside it could
    # leave one running with no handle to reach it by: a signal here is only
    # noted, and replayed once the child is covered by the cleanup below.
    pending: list[int] = []
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous:
        signal.signal(sig, lambda signum, frame: pending.append(signum))
    job: _TreeJob | None = None
    if sys.platform == "win32":
        try:
            job = _TreeJob.enter()  # before the spawn: pytest is born inside it
        except OSError as exc:
            sys.stderr.write(
                f"run_test_benchmark: no job object for the pytest tree ({exc}); "
                "a stop reaches only what taskkill /T can still find\n"
            )
        record["tree_job"] = job is not None
    try:
        try:
            proc = subprocess.Popen(command, **popen_kwargs)  # fixed argv, no shell
        except OSError as exc:
            sys.stderr.write(
                f"run_test_benchmark: cannot start pytest: {type(exc).__name__}: {exc}\n"
            )
            finish("spawn_failed", None, error=type(exc).__name__)
            _summarize(root)
            return EXIT_SPAWN_FAILED
        try:
            record["child_pid"] = proc.pid
            # Installed inside the guard: a signal landing as the first raising
            # handler goes in must still reach the tree kill below.
            for sig in previous:
                signal.signal(sig, _raise_interrupted)
            if pending:
                raise _Interrupted(pending[0])
            _write_runner(root, record)
            returncode = _wait_for_exit(proc, deadline_mono)
            # pytest is done. Hold signals until the terminal record is written:
            # the ``finally`` below still has the raising handler installed while
            # it closes the job, and restores the caller's before this record
            # would otherwise be written, so a late cancel would escape ``main``
            # with the record at ``running``. One that lands before the hold is
            # still caught below.
            _hold_signals_for_cleanup()
            finish("exited", returncode)
        except subprocess.TimeoutExpired:
            _hold_signals_for_cleanup()
            sys.stderr.write(
                f"run_test_benchmark: deadline reached after {budget:.0f}s; stopping pytest\n"
            )
            _kill_tree(proc, args.grace, job)
            finish("timed_out", proc.returncode, child_reaped=proc.returncode is not None)
            _summarize(root)
            return EXIT_TIMED_OUT
        except _Interrupted:
            _hold_signals_for_cleanup()
            if sys.platform == "win32":
                # A console Ctrl+C already reached pytest too (same console
                # group): give it the grace to finish before the tree kill.
                try:
                    proc.wait(timeout=args.grace)
                except subprocess.TimeoutExpired:
                    pass
            _kill_tree(proc, args.grace, job)
            finish("interrupted", proc.returncode, child_reaped=proc.returncode is not None)
            _summarize(root)
            return EXIT_INTERRUPTED
        except BaseException:
            # Any other fault once the child exists (a runner.json write on a
            # full metrics volume, say) must not orphan it: pytest sits in a
            # session of its own on POSIX and the job has no kill-on-close on
            # Windows, so nothing else would stop the tree. The fault still
            # propagates; the record stays at ``running``, which no verifier
            # accepts.
            _hold_signals_for_cleanup()
            _kill_tree(proc, args.grace, job)
            raise
    finally:
        if job is not None:
            job.close()  # no kill-on-close: this frees the handle, nothing more
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    _summarize(root)
    if returncode < 0:
        return 128 - returncode
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
