"""Cursor driver: sessions run inside a Node sidecar over the ``@cursor/sdk`` API.

Cursor's headless agent is a **Node** library, not a terminal program: there is
no CLI to inject a prompt into and no hook dialect to register, so this family
cannot be a tmux profile. It is instead an adapter *kind* — registered in
:mod:`~.registry` as ``cursor-sdk`` with ``needs_mux = False`` and selected by
the packaged ``cursor-sdk`` profile's ``adapter`` field — that spawns one
short-lived Node process per session (``data/cursor-sidecar.mjs``) and reads its
stdout.

Transport contract (the sidecar is ours, so this is a contract we own, not one
probed off a third-party binary):

- The sidecar writes **NDJSON on stdout**: every ``@cursor/sdk`` stream event as
  one JSON object per line, then exactly one final *sentinel* object
  ``{"type": "__sidecar_result__", …}``. The sentinel is the completion signal —
  the analogue of a ``Stop`` hook — and carries ``status`` (the SDK run status),
  ``agentId`` / ``runId``, and the SDK's ``usage`` object.
- Anything the sidecar itself fails on (a missing ``@cursor/sdk``, a rejected
  API key) still ends in a sentinel, with ``status: "error"``.
- **stderr is a separate file**, ``logs/<task_id>.sidecar.err``, and that is what
  :class:`~.env_fault.EnvFaultMixin` scans. ``logs/<task_id>.log`` carries the
  event stream and therefore the model's own words, so it is not a sound target
  for content-anchored patterns — the same split ``opencode_http`` makes between
  its transcript and ``.server.out``.
- One run per session. ``@cursor/sdk`` has no mid-run injection channel
  (``agent.send`` starts a *new* run), so this family sends no wake-nudges and no
  contract nudge; see :class:`CursorSdkDevAdapter`.

The ``@cursor/sdk`` runtime is not a Python dependency and cannot be one. It is
installed on demand by ``bmad-loop init --provision cursor-sdk``
(:func:`provision_sdk`), into :func:`sdk_home` — by default
``~/.bmad-loop/cursor-sdk``. ``validate`` reports the three preconditions
(:func:`validate_environment`).
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from .. import envvars, runs
from ..journal import LOGS_DIR
from ..model import TokenUsage
from ..policy import Policy
from .base import CodingCLIAdapter, SessionHandle, SessionResult, SessionSpec
from .env_fault import EnvFaultMixin
from .generic import (
    HEARTBEAT_INTERVAL_S,
    PROOF_OF_WORK_MIN_LOG_BYTES,
    _DevSynthesisMixin,
    _ResultFileMixin,
)
from .profile import CLIProfile
from .registry import ProvisionError

if TYPE_CHECKING:
    from ..bmadconfig import ProjectPaths

#: The ``profile.adapter`` value this family registers under. Duplicated as
#: ``registry.CURSOR_SDK`` so ``validate`` can key on the kind without importing
#: this module (and with it a Node-shaped provisioning path) to do it.
PROVIDER = "cursor-sdk"
#: Model used when neither the role's policy nor the run sets one. The SDK
#: requires an explicit model id, so unlike a CLI there is no "server default".
DEFAULT_MODEL = "composer-2.5"
#: ``@cursor/sdk`` needs the Node 22 LTS line; 22.13 is the floor its own
#: package metadata declares.
MIN_NODE = (22, 13)
#: Version range pinned into the generated ``package.json``. A range rather than
#: an exact pin: the runtime is installed on the operator's machine outside
#: ``uv.lock``, so patch fixes should land without a bmad-loop release.
SDK_PIN = "^1.0.23"
#: The sidecar's terminal object; see the module docstring.
SENTINEL_TYPE = "__sidecar_result__"
#: How long to keep polling the sidecar's stdout queue per loop tick. Short
#: enough that a hard-stop request is noticed promptly, like the other adapters.
POLL_TICK_S = 1.0
#: Grace after a clean sentinel for the session's on-disk artifact to be flushed.
RESULT_GRACE_S = 15.0
#: How long to wait for the sidecar to exit after SIGTERM before killing it.
KILL_WAIT_S = 5.0
#: npm install budget for :func:`provision_sdk`.
PROVISION_TIMEOUT_S = 300.0
_NODE_VERSION_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")


class CursorSdkError(Exception):
    """A cursor-sdk session could not be spawned or driven.

    Carried by this family's :class:`~.registry.AdapterBuilder` as its
    ``construct_error``, so a construction failure becomes a clean ``SystemExit``
    from ``runsetup.make_adapters`` instead of a traceback.

    Provisioning failures are NOT this type: they are the seam's
    :class:`~.registry.ProvisionError`, re-exported here, because the installer
    reports any family's provisioning failure without importing that family."""


def sdk_home() -> Path:
    """Where the provisioned ``@cursor/sdk`` runtime lives.

    Machine-scoped rather than project-scoped: the runtime is a Node install,
    identical for every project on the host, and putting it in the project tree
    would put ``node_modules`` inside a repo the sessions themselves commit to.
    """
    override = envvars.cursor_sdk_dir()
    return Path(override).expanduser() if override else Path.home() / ".bmad-loop" / PROVIDER


def _sdk_package_json() -> Path:
    return sdk_home() / "node_modules" / "@cursor" / "sdk" / "package.json"


def parse_node_version(text: str) -> tuple[int, int, int] | None:
    """``(major, minor, patch)`` out of ``node --version`` output, or None."""
    match = _NODE_VERSION_RE.search(text.strip())
    if match is None:
        return None
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def parse_usage(sentinel: dict[str, Any] | None) -> TokenUsage | None:
    """The SDK ``usage`` object as a :class:`TokenUsage`, or None when absent.

    Every field degrades to 0 independently: the SDK's usage shape has grown
    keys over time, and a missing or non-numeric one must cost that column
    rather than the whole tally."""
    usage = (sentinel or {}).get("usage")
    if not isinstance(usage, dict):
        return None

    def integer(key: str) -> int:
        try:
            return int(usage.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    return TokenUsage(
        input_tokens=integer("inputTokens"),
        output_tokens=integer("outputTokens"),
        cache_read_tokens=integer("cacheReadTokens"),
        cache_creation_tokens=integer("cacheWriteTokens"),
    )


def validate_environment(binary: str) -> tuple[list[str], list[str]]:
    """The three cursor-sdk preconditions, as ``(notes, problems)`` for
    ``validate``: a new-enough Node, a provisioned ``@cursor/sdk``, and an API
    key. Reported at ``validate`` time because each fails a run at *launch*,
    where the only symptom is a session that dies before doing anything."""
    notes: list[str] = []
    problems: list[str] = []
    floor = f"{MIN_NODE[0]}.{MIN_NODE[1]}"
    node = shutil.which(binary)
    if node is None:
        problems.append(f"{binary!r} not found on PATH — {PROVIDER} needs Node >= {floor}")
    elif (version := _node_version(node)) is None:
        problems.append(f"`{binary} --version` did not report a version — {PROVIDER} needs Node >= {floor}")
    elif version < MIN_NODE:
        problems.append(f"node {'.'.join(map(str, version))} is below the {floor} floor for {PROVIDER}")
    else:
        notes.append(f"node {'.'.join(map(str, version))} found ({node})")
    home = sdk_home()
    if _sdk_package_json().is_file():
        notes.append(f"@cursor/sdk present at {home}")
    else:
        problems.append(
            f"@cursor/sdk not found under {home}/node_modules — "
            f"run `bmad-loop init --provision {PROVIDER}`"
        )
    if os.environ.get("CURSOR_API_KEY"):
        notes.append("CURSOR_API_KEY is set")
    else:
        problems.append("CURSOR_API_KEY is not set")
    return notes, problems


def _node_version(node: str) -> tuple[int, int, int] | None:
    try:
        completed = subprocess.run(  # noqa: S603 — resolved binary, fixed argv
            [node, "--version"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_node_version(completed.stdout)


def provision_sdk() -> list[str]:
    """Install ``@cursor/sdk`` into :func:`sdk_home`, returning progress notes.

    Reached only from ``bmad-loop init --provision cursor-sdk``, never from a
    run: it touches the network, so it is opt-in and human-triggered. Idempotent
    — an already-provisioned home is reported and left alone. Raises
    :class:`ProvisionError` on any failure; the installer turns that into a
    ``FAIL:`` line and rc 1 rather than a traceback."""
    home = sdk_home()
    if _sdk_package_json().is_file():
        return [f"@cursor/sdk already provisioned at {home}"]
    npm = shutil.which("npm")
    if npm is None:
        raise ProvisionError(
            f"npm not found on PATH; install Node >= {MIN_NODE[0]}.{MIN_NODE[1]} "
            f"and re-run `bmad-loop init --provision {PROVIDER}`"
        )
    home.mkdir(parents=True, exist_ok=True)
    (home / "package.json").write_text(
        json.dumps(
            {
                "name": "bmad-loop-cursor-sdk",
                "private": True,
                "type": "module",
                "dependencies": {"@cursor/sdk": SDK_PIN},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        completed = subprocess.run(  # noqa: S603 — resolved npm, fixed argv
            [npm, "install", "--no-audit", "--no-fund"],
            cwd=home,
            capture_output=True,
            text=True,
            timeout=PROVISION_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ProvisionError(f"could not install @cursor/sdk in {home}: {error}") from error
    if completed.returncode or not _sdk_package_json().is_file():
        detail = (completed.stderr or completed.stdout).strip()[-1000:]
        raise ProvisionError(f"npm install failed in {home}: {detail}")
    return [f"@cursor/sdk installed at {home}"]


@dataclass
class _Sidecar:
    """One session's Node process and the stdout lines its pump has read."""

    proc: subprocess.Popen[str] | None
    lines: queue.Queue[str | None]
    err_fh: IO[str] | None = None
    sentinel: dict[str, Any] | None = None
    spawn_error: str | None = None


class CursorSdkAdapter(_ResultFileMixin, EnvFaultMixin, CodingCLIAdapter):
    # The sidecar's own stderr, NOT <task_id>.log — that file is the SDK event
    # stream and carries the model's output, so content-anchored patterns
    # matched against it would fire on a story that merely writes about a
    # provider error. Same split, same reason, as opencode-http's `.server.out`.
    ENV_FAULT_LOG_SUFFIX = ".sidecar.err"

    injection = "launch-flag"
    observation = "stream"
    state = "local-json-tree"

    def __init__(
        self,
        run_dir: Path,
        policy: Policy,
        profile: CLIProfile,
        binary: str | None = None,
        extra_args: tuple[str, ...] | None = None,
        usage_grace_s: float | None = None,
        stop_without_result_nudges: int | None = None,
        events_dir: Path | None = None,
    ):
        # Three of the bootstrap keywords are accepted and unused, like the
        # opencode family's `events_dir`. This transport fires no hooks (no event
        # channel to point at); its usage arrives in-band with the completion
        # sentinel (no transcript to wait on, so no grace); and a run is one
        # `agent.send` with no way to talk to a live turn, so a result-less
        # turn-end cannot be nudged into producing one — zero is the honest
        # budget, not a conservative default. They are part of the run
        # description `runsetup.make_adapters` hands every family, and refusing a
        # kwarg here would make that bootstrap branch per family.
        del events_dir, usage_grace_s, stop_without_result_nudges
        self.run_dir = run_dir
        self.policy = policy
        self.profile = profile
        self.name = profile.name
        self.binary = binary or profile.binary
        self.extra_args = extra_args
        self._stop_nudges = 0
        self.result_grace_s = RESULT_GRACE_S
        self.kill_wait_s = KILL_WAIT_S
        self.poll_tick_s = POLL_TICK_S
        self.sdk_home = sdk_home()
        self.tasks_dir = run_dir / "tasks"
        self.logs_dir = run_dir / LOGS_DIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._sidecars: dict[str, _Sidecar] = {}
        self._usage: dict[str, TokenUsage] = {}

    # ------------------------------------------------------------- spawning

    def _ensure_sidecar_script(self) -> Path:
        """Materialize the packaged sidecar into the SDK home, where it can
        resolve ``@cursor/sdk`` from the sibling ``node_modules``. Rewritten
        whenever the packaged bytes differ, so an upgraded bmad-loop does not
        keep driving the previous release's sidecar."""
        target = self.sdk_home / "cursor-sidecar.mjs"
        want = resources.files("bmad_loop.data").joinpath("cursor-sidecar.mjs").read_bytes()
        self.sdk_home.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_bytes() != want:
            target.write_bytes(want)
        return target

    def _sidecar_argv(self, spec: SessionSpec, script: Path, prompt_file: Path) -> list[str]:
        """argv for one session. A seam: tests point it at a scripted fake
        sidecar so the transport is exercised without Node or an API key."""
        return [
            self.binary,
            str(script),
            "--cwd",
            str(spec.cwd),
            "--model",
            spec.model or DEFAULT_MODEL,
            "--prompt-file",
            str(prompt_file),
            "--timeout-ms",
            str(int(spec.timeout_s * 1000)),
            *(self.extra_args or ()),
        ]

    def start_session(self, spec: SessionSpec) -> SessionHandle:
        task_dir = self.tasks_dir / spec.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        rendered = self.profile.render_prompt(spec.prompt)
        prompt_file = task_dir / "prompt.txt"
        prompt_file.write_text(rendered + "\n", encoding="utf-8")
        # Any result from a previous attempt on this task id would otherwise read
        # as this session's, exactly as the tmux adapters guard against. This
        # unlink is what makes the read-back below authoritative: the path is
        # task-scoped, so anything at it afterwards was written by THIS session.
        self._result_path(spec.task_id).unlink(missing_ok=True)
        # Created empty before the process exists, for the same reason the tmux
        # adapter pre-creates its pane log: a session that dies on arrival must
        # report "rendered nothing" (False) to the proof-of-work gate rather than
        # "no such signal" (None), which the gate treats as inert.
        (self.logs_dir / f"{spec.task_id}.log").touch()
        launched_ns = time.time_ns()
        lines: queue.Queue[str | None] = queue.Queue()
        err_fh: IO[str] | None = None
        try:
            argv = self._sidecar_argv(spec, self._ensure_sidecar_script(), prompt_file)
            err_fh = (self.logs_dir / f"{spec.task_id}{self.ENV_FAULT_LOG_SUFFIX}").open(
                "a", encoding="utf-8"
            )
            proc = subprocess.Popen(  # noqa: S603 — argv from the profile + run config
                argv,
                cwd=spec.cwd,
                # Same merge and same pin chokepoint as the tmux window: the
                # profile's [env] table must not move the sidecar off this
                # process's state root (`runs.pin_state_root`).
                env={**os.environ, **runs.pin_state_root({**self.profile.env, **spec.env})},
                stdout=subprocess.PIPE,
                stderr=err_fh,
                text=True,
                bufsize=1,
            )
        except (OSError, ValueError) as error:
            if err_fh is not None:
                err_fh.close()
            self._note_lifecycle(spec.task_id, "sidecar-spawn-failed", error=str(error))
            self._sidecars[spec.task_id] = _Sidecar(None, lines, spawn_error=str(error))
            lines.put(None)
            return SessionHandle(spec.task_id, "spawn-failed", launched_ns)
        sidecar = _Sidecar(proc, lines, err_fh=err_fh)
        self._sidecars[spec.task_id] = sidecar
        threading.Thread(
            target=self._pump,
            args=(sidecar, self.logs_dir / f"{spec.task_id}.log"),
            daemon=True,
        ).start()
        return SessionHandle(spec.task_id, str(proc.pid), launched_ns)

    @staticmethod
    def _pump(sidecar: _Sidecar, log_path: Path) -> None:
        """Tee the sidecar's stdout to the session log and the wait loop's
        queue. Always terminates the queue with a ``None``, whatever went wrong,
        so the loop can never block on a dead reader."""
        try:
            proc = sidecar.proc
            assert proc is not None and proc.stdout is not None
            with log_path.open("a", encoding="utf-8") as log:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    sidecar.lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            sidecar.lines.put(None)

    # ------------------------------------------------------------- the wait

    def wait_for_completion(self, handle: SessionHandle, spec: SessionSpec) -> SessionResult:
        """Drain the sidecar's stdout until the stream ends, then rule on what
        the sentinel said.

        Unlike a hook transport there is no ambiguity about turn-end: the
        sidecar emits exactly one sentinel and then exits, so stdout EOF is
        proof the run is over. The verdict is therefore only ever three-way —
        the deadline fired, the sentinel said ``finished``, or it did not."""
        sidecar = self._sidecars.get(handle.task_id)
        if sidecar is None or sidecar.spawn_error is not None:
            reason = (
                sidecar.spawn_error
                if sidecar is not None and sidecar.spawn_error is not None
                else "no sidecar was started"
            )
            self._note_resultless_stop(handle.task_id, "sidecar-never-ran", reason)
            return self._final(handle, spec, "crashed", None, None)
        deadline = time.monotonic() + spec.timeout_s
        # Wall-clock co-bound (#157): a host suspend freezes time.monotonic(),
        # silently extending the monotonic deadline by the nap's length. The
        # wall clock keeps counting through a suspend, so it may EXPIRE the
        # deadline — never extend it.
        wall_deadline = time.time() + spec.timeout_s
        last_heartbeat: float | None = None
        eof = False

        while not eof:
            remaining = deadline - time.monotonic()
            wall_expired = time.time() >= wall_deadline
            if remaining <= 0 or wall_expired:
                if remaining <= 0 and wall_expired:
                    expired = "both"
                elif remaining <= 0:
                    expired = "monotonic"
                else:
                    # wall-only expiry with monotonic time to spare: the
                    # monotonic clock stood still — the suspend signature.
                    expired = "wall"
                self._note_lifecycle(
                    handle.task_id,
                    "timeout-fired",
                    expired_clock=expired,
                    timeout_s=spec.timeout_s,
                    mono_remaining_s=round(remaining, 3),
                )
                self._terminate(sidecar)
                return SessionResult(
                    status="timeout",
                    session_id=self._session_id(sidecar),
                    timeout_fired_at=time.time(),
                    timeout_expired_clock=expired,
                )
            # Hard-stop poll (#319). Return the verdict — never raise
            # `RunStopped` here: that would skip `run()`'s finally-kill. The
            # request file is left on disk for the engine to attribute the stop.
            if self._hard_stop_requested():
                self._note_lifecycle(handle.task_id, "stop-abort-fired")
                self._terminate(sidecar)
                return SessionResult(status="aborted", session_id=self._session_id(sidecar))
            now = time.monotonic()
            if last_heartbeat is None or now - last_heartbeat >= HEARTBEAT_INTERVAL_S:
                last_heartbeat = now
                self._write_heartbeat(
                    handle.task_id,
                    {"ts": time.time(), "remaining_s": round(remaining, 3)},
                )
                # Mid-session spec-status transition sampling (#276 M2) rides the
                # heartbeat cadence — inert on this class, live on the dev
                # subclass, exactly as on the other two families.
                self._observe_tick(handle, spec)
            try:
                line = sidecar.lines.get(timeout=min(self.poll_tick_s, max(remaining, 0.0)))
            except queue.Empty:
                continue
            if line is None:
                eof = True
                continue
            self._absorb(sidecar, line)

        sentinel = sidecar.sentinel
        session_id = self._session_id(sidecar)
        usage = parse_usage(sentinel)
        if session_id is not None and usage is not None:
            self._usage[session_id] = usage
        finished = sentinel is not None and sentinel.get("status") == "finished"
        if not finished:
            # No turn ever ended, so this is a crash — but the read-back still
            # runs, exactly as it does on the tmux adapters' window-death path. A
            # skill that wrote its result and *then* lost the SDK run did the
            # work, and `_final` will only honor an artifact the launch-time
            # unlink proves belongs to this session.
            detail = "no sentinel" if sentinel is None else f"status={sentinel.get('status')!r}"
            self._note_resultless_stop(handle.task_id, "sidecar-not-finished", detail)
            return self._final(handle, spec, "crashed", session_id, None)
        # A clean sentinel is this transport's `Stop`: the turn genuinely ended,
        # so the artifact read-back may wait out the flush window, and a session
        # that produced none is a stall rather than a crash.
        result_json = self._result_json(handle, spec, wait=True)
        if result_json is None:
            self._note_resultless_stop(handle.task_id, "sentinel-without-result", "")
        return SessionResult(
            status="completed" if result_json is not None else "stalled",
            result_json=result_json,
            session_id=session_id,
            stop_seen=True,
        )

    def _absorb(self, sidecar: _Sidecar, line: str) -> None:
        """Record the sentinel if this stdout line is one. Every other line is
        already tee'd to the session log by the pump and needs nothing here; a
        line that is not JSON at all (a Node warning that reached stdout) is
        skipped rather than treated as a fault."""
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if isinstance(event, dict) and event.get("type") == SENTINEL_TYPE:
            sidecar.sentinel = event

    def _log_evidence(self, handle: SessionHandle) -> bool | None:
        """Session-log half of the #261 proof-of-work gate (see
        ``_ResultFileMixin._produced_work``). The pump appends every stdout line
        to ``logs/<task_id>.log``, so its size measures how much the run emitted.

        Same tristate contract as the tmux adapter's pane-log version: True past
        the byte floor, False for a log that exists and stayed empty (a sidecar
        that died on arrival — the case the gate exists for, pre-created in
        ``start_session`` so it cannot read as unknown), None for a handle this
        adapter never launched."""
        try:
            size = (self.logs_dir / f"{handle.task_id}.log").stat().st_size
        except OSError:
            return None
        return size > PROOF_OF_WORK_MIN_LOG_BYTES

    @staticmethod
    def _session_id(sidecar: _Sidecar) -> str | None:
        sentinel = sidecar.sentinel
        if sentinel is None:
            return None
        raw = sentinel.get("runId") or sentinel.get("agentId")
        return str(raw) if raw else None

    # ------------------------------------------------------------- teardown

    def _terminate(self, sidecar: _Sidecar) -> None:
        proc = sidecar.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=self.kill_wait_s)
            except subprocess.TimeoutExpired:
                proc.kill()
        if sidecar.err_fh is not None:
            try:
                sidecar.err_fh.close()
            except OSError:
                pass
            sidecar.err_fh = None

    def kill(self, handle: SessionHandle) -> None:
        sidecar = self._sidecars.get(handle.task_id)
        if sidecar is not None:
            self._terminate(sidecar)

    def read_usage(self, result: SessionResult) -> TokenUsage | None:
        return self._usage.get(result.session_id) if result.session_id else None


class CursorSdkDevAdapter(_DevSynthesisMixin, CursorSdkAdapter):
    """Dev/review adapter for the generic ``bmad-build-auto`` skill on the SDK.

    That skill writes NO ``result.json`` — its outcome lives in the terminal
    spec it leaves on disk, which :class:`_DevSynthesisMixin` locates and
    synthesizes via :mod:`devcontract`, the same machinery the tmux and HTTP
    families share.
    """

    def __init__(self, *args, paths: ProjectPaths, **kwargs):
        super().__init__(*args, **kwargs)
        self.paths = paths
        self._configure_dev_knobs()
        # `_configure_dev_knobs` arms the stall grace and the two nudge budgets
        # for a transport that can talk to a live turn. This one cannot: a
        # `@cursor/sdk` run ends when the sidecar's sentinel lands, and the only
        # way to say anything further is to start a new run — which would be a
        # second session, not a nudge. So the budgets are emptied rather than
        # left to fire against a channel that does not exist.
        self._stall_grace_s = 0.0
        self._stall_nudges = 0
        self._contract_nudge_enabled = False

    def _probe_alive(self, handle: SessionHandle) -> bool | None:
        sidecar = self._sidecars.get(handle.task_id)
        if sidecar is None or sidecar.proc is None:
            return False  # never spawned: nothing we own is alive
        # Never None: the live Popen handle pins the pid, so poll() is always
        # answerable — unlike a tmux transport there is no probe that can hang.
        return sidecar.proc.poll() is None
