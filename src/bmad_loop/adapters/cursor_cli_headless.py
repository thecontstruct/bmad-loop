"""Headless Cursor CLI adapter — ``cursor-agent -p`` supervised as a subprocess.

Cursor's CLI has no interactive pane worth driving and fires no lifecycle hooks,
so neither the tmux-injection nor the hook-signal half of the generic transport
applies. It does expose a fully typed one-shot mode — ``cursor-agent -p
--output-format stream-json`` — which streams NDJSON frames to stdout and exits
when the turn ends. This adapter supervises exactly that: one child process per
session, no terminal multiplexer, no hook config (``needs_mux=False``, and the
profile is ``[hooks] dialect = "none"``).

**Completion-path invariant.** AGENTS.md allows a session to complete only on a
hook Stop event or window death, and this transport keeps both halves rather
than inventing a third:

- the terminal ``{"type": "result"}`` frame ≙ the **Stop hook**. It is a typed
  control frame the CLI emits itself, not model prose — nothing the model writes
  can forge it, because a model turn is carried in ``assistant``/``text`` frames
  that this loop parses and ignores.
- the child process exiting (stdout EOF) ≙ **window death**, with the same
  landed-artifact trust the tmux crash path gives it.

Neither ends a session on its own say-so: the verdict is finalized by
:meth:`_ResultFileMixin._final`, so the result is whatever the on-disk artifact
read-back proves, and ``stop_seen`` carries the result-frame sighting into the
#261 proof-of-work gate exactly as a real Stop would. The frame's own
``is_error``/``subtype`` self-report is recorded as a lifecycle breadcrumb and
never read as a verdict — that would be the same forbidden path, inverted.

**No mid-turn injection.** ``-p`` is one-shot: once launched, the turn cannot be
nudged, so :meth:`send_text` stays unimplemented and both nudge budgets are
pinned to zero (see :meth:`CursorCliHeadlessDevAdapter.__init__`). A stalled
session runs out its clock instead of being woken — an honest limit of the
transport, not something to fake.

**Two log sinks, one per audience.** ``logs/<task-id>.log`` takes the raw
stream-json stdout (the transcript — it carries the model's own words) and
``logs/<task-id>.err`` takes the CLI's stderr, which is where auth failures and
transport errors land. Env-fault classification points ``ENV_FAULT_LOG_SUFFIX``
at ``.err`` for the same safety reason ``opencode_http`` points it at
``.server.out``: a pattern matched against a log the model can write to is
unsound, because a story that quotes a provider error is byte-identical to the
real thing. The shipped profile declares no ``env_fault_patterns``, so
classification is inert today; the sink split is what makes adding one later
sound.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..journal import LOGS_DIR
from ..model import TokenUsage
from .base import CodingCLIAdapter, SessionHandle, SessionResult, SessionSpec
from .env_fault import EnvFaultMixin
from .generic import HEARTBEAT_INTERVAL_S, _DevSynthesisMixin, _ResultFileMixin
from .profile import CLIProfile

if TYPE_CHECKING:
    from ..bmadconfig import ProjectPaths
    from ..policy import Policy

PROVIDER = "cursor-cli-headless"
BINARY = "cursor-agent"
#: Max stdout-queue wait per loop tick. Bounds how long one iteration sits
#: between the two hard-stop polls, matching the generic/opencode cadence.
POLL_TICK_S = 5.0
#: Extra wall time granted past ``spec.timeout_s`` for the child to flush its
#: final frames and exit after the turn itself has ended.
WAIT_SLACK_S = 30.0
#: How long to wait for the child to exit on its own after stdout reaches EOF,
#: before falling back to terminate/kill.
EXIT_GRACE_S = 5.0
#: Sentinel pushed by the reader thread when stdout reaches EOF. A plain
#: ``None`` would be ambiguous with "the queue wait expired".
_EOF = object()
#: ``result``-frame subtypes that report a turn the CLI did not itself call a
#: failure. A real success frame is ``subtype: "success"`` (observed on
#: cursor-agent 2026.08.04); an absent subtype says nothing either way.
SUCCESS_SUBTYPES = frozenset({"success"})
#: Cap on the frame's own ``result`` string when it is copied into a breadcrumb.
RESULT_DETAIL_MAX_CHARS = 500


def build_argv(
    *,
    prompt: str,
    cwd: Path,
    model: str = "",
    binary: str = BINARY,
    bypass: tuple[str, ...] = (),
) -> list[str]:
    """argv for one headless turn.

    ``-p`` + ``--output-format stream-json`` are structural — they are what makes
    this transport observable at all — so they are built here rather than left to
    the profile. ``bypass`` is the profile's ``bypass_args`` (or the policy
    ``extra_args`` that replace them), which is where the approval/trust flags
    live so an operator can override them without editing Python. The prompt is
    positional and must stay last."""
    argv = [binary, "-p", *bypass, "--output-format", "stream-json", "--workspace", str(cwd)]
    if model.strip():
        argv.extend(["--model", model.strip()])
    return [*argv, prompt]


def parse_usage(event: dict[str, Any] | None) -> TokenUsage | None:
    """Token counts off the terminal ``result`` frame's ``usage`` object.

    ``reasoningTokens`` is deliberately NOT added to output, diverging from the
    copilot and gemini parsers in :mod:`bmad_loop.tokens`, which fold their
    vendors' reasoning/thoughts counts in under the identical key spelling.
    Cursor documents this object's ``reasoningTokens`` as *a subset of*
    ``outputTokens`` (``@cursor/sdk`` ``usage-types.d.ts``: "``totalTokens``
    excludes ``reasoningTokens`` (a subset of output)"), and a real run's
    input + output + cache reads + cache writes equalled the reported
    ``totalTokens`` exactly. Adding it here would double-count. Same spelling as
    copilot's field, different semantics — do not "fix" this back."""
    usage = (event or {}).get("usage")
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


def result_failure(event: dict[str, Any] | None) -> dict[str, Any] | None:
    """The terminal ``result`` frame's own failure self-report, or None.

    The frame carries ``is_error`` and ``subtype`` beside a short ``result``
    string, and the adapter used to drop all three — leaving a failed turn
    byte-identical to a successful one until downstream verification happened to
    fail it. This is a **diagnostic read only**: the caller records what it
    returns and nothing else. The session verdict stays artifact-derived (see
    :meth:`_ResultFileMixin._final`), because taking a verdict off a
    self-reported frame is the completion-on-self-report path AGENTS.md forbids,
    run in the other direction.

    A subtype outside :data:`SUCCESS_SUBTYPES` counts as a failure report. The
    crumb changes no outcome, so an over-eager one costs a log line while a
    missed one costs a session nobody can diagnose."""
    if not isinstance(event, dict):
        return None
    is_error = bool(event.get("is_error"))
    raw_subtype = event.get("subtype")
    subtype = raw_subtype.strip() if isinstance(raw_subtype, str) else ""
    if not is_error and (not subtype or subtype in SUCCESS_SUBTYPES):
        return None
    detail: dict[str, Any] = {"subtype": subtype, "is_error": is_error}
    text = event.get("result")
    if isinstance(text, str) and text.strip():
        # The frame's own one-line outcome, capped. Nothing else off the stream is
        # copied: `.log` is the model's stream, this breadcrumb file is not.
        detail["result"] = text.strip()[:RESULT_DETAIL_MAX_CHARS]
    return detail


@dataclass
class _Running:
    """Per-session child-process state. ``proc`` is None only when the spawn
    itself failed, in which case ``spawn_error`` carries why."""

    proc: subprocess.Popen[str] | None
    lines: queue.Queue[str | object]
    result_event: dict[str, Any] | None = None
    spawn_error: str | None = None
    sinks: list[Any] = field(default_factory=list)


class CursorCliHeadlessAdapter(_ResultFileMixin, EnvFaultMixin, CodingCLIAdapter):
    """Plain (non-synthesizing) headless Cursor adapter: the skill writes
    ``tasks/<task_id>/result.json`` and :class:`_ResultFileMixin` reads it back."""

    # See the module docstring: `.log` is the model's own stream, `.err` is not.
    ENV_FAULT_LOG_SUFFIX = ".err"

    injection = "launch-flag"
    observation = "stream-json"
    state = "local-json-tree"

    def __init__(
        self,
        run_dir: Path,
        policy: "Policy",
        profile: CLIProfile,
        binary: str | None = None,
        extra_args: tuple[str, ...] | None = None,
        usage_grace_s: float | None = None,
        stop_without_result_nudges: int | None = None,
        events_dir: Path | None = None,
    ) -> None:
        # `events_dir` is accepted and unused for the same reason opencode_http
        # accepts it: this family fires no hooks, so it has no event channel to
        # point at, but it is part of the run description the bootstrap hands
        # every family (#494) and refusing it would make the bootstrap branch.
        # `usage_grace_s` likewise: usage rides the result frame, so there is no
        # transcript to keep re-reading after the session ends.
        del events_dir, usage_grace_s
        # No nudge budget is meaningful here — `-p` cannot be injected into
        # mid-turn, so a Stop-without-result is terminal whatever the profile or
        # policy asks for. Pinned to 0 rather than read from either.
        del stop_without_result_nudges
        self.run_dir = run_dir
        self.policy = policy
        self.profile = profile
        self.name = profile.name
        self.binary = binary or profile.binary
        # Same precedence every other adapter uses: an explicit policy
        # `extra_args` REPLACES the profile's bypass flags; None means "unset",
        # which is not the same as an empty tuple (that means "no flags").
        self.bypass_args = tuple(extra_args if extra_args is not None else profile.bypass_args)
        self._stop_nudges = 0
        self._stall_grace_s = 0.0
        self._stall_nudges = 0
        self.wait_slack_s = WAIT_SLACK_S
        self.poll_tick_s = POLL_TICK_S
        self.exit_grace_s = EXIT_GRACE_S
        self.tasks_dir = run_dir / "tasks"
        self.logs_dir = run_dir / LOGS_DIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._running: dict[str, _Running] = {}
        self._usage: dict[str, TokenUsage] = {}

    # ------------------------------------------------------------- lifecycle

    def start_session(self, spec: SessionSpec) -> SessionHandle:
        task_dir = self.tasks_dir / spec.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        # The engine hands every adapter the canonical "/skill args" prompt; it is
        # the adapter's job to run it through the profile template (see
        # GenericAdapter.build_command / OpencodeHttpAdapter.start_session).
        rendered = self.profile.render_prompt(spec.prompt)
        (task_dir / "prompt.txt").write_text(rendered + "\n", encoding="utf-8")
        # Unlink before launch so a stale result.json from a previous attempt can
        # never be read back as this session's work (the property `_ResultFileMixin`
        # relies on to skip the proof-of-work gate on its own read-back).
        (task_dir / "result.json").unlink(missing_ok=True)
        lines: queue.Queue[str | object] = queue.Queue()
        launched_ns = time.time_ns()
        argv = build_argv(
            prompt=rendered,
            cwd=spec.cwd,
            model=spec.model,
            binary=self.binary,
            bypass=self.bypass_args,
        )
        try:
            err_sink = (self.logs_dir / f"{spec.task_id}.err").open("a", encoding="utf-8")
        except OSError:
            err_sink = None
        try:
            proc = subprocess.Popen(  # noqa: S603 — argv list, no shell
                argv,
                cwd=spec.cwd,
                env={**os.environ, **spec.env},
                stdout=subprocess.PIPE,
                stderr=err_sink or subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (OSError, ValueError) as error:
            if err_sink is not None:
                err_sink.close()
            running = _Running(None, lines, spawn_error=str(error))
            lines.put(_EOF)
            self._running[spec.task_id] = running
            self._note_lifecycle(spec.task_id, "spawn-failed", error=str(error))
            return SessionHandle(spec.task_id, "spawn-failed", launched_ns)
        running = _Running(proc, lines, sinks=[err_sink] if err_sink else [])
        threading.Thread(
            target=self._pump,
            args=(running, self.logs_dir / f"{spec.task_id}.log"),
            daemon=True,
        ).start()
        self._running[spec.task_id] = running
        return SessionHandle(spec.task_id, str(proc.pid), launched_ns)

    @staticmethod
    def _pump(running: _Running, log_path: Path) -> None:
        """Tee the child's stdout to the transcript and hand each line to the
        wait loop. Best-effort: an unwritable log must not wedge the session, and
        the ``finally`` guarantees the loop always sees EOF."""
        try:
            proc = running.proc
            if proc is None or proc.stdout is None:
                return
            with log_path.open("a", encoding="utf-8") as log:
                for line in proc.stdout:
                    try:
                        log.write(line)
                        log.flush()
                    except OSError:
                        pass
                    running.lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            running.lines.put(_EOF)

    def wait_for_completion(self, handle: SessionHandle, spec: SessionSpec) -> SessionResult:
        running = self._running.get(handle.task_id)
        if running is None or running.spawn_error is not None:
            # The binary never started: nothing ran, so nothing may be read back.
            return SessionResult(status="crashed")
        # Wall-clock co-bound (#157): a host suspend freezes time.monotonic() and
        # would silently extend the deadline. The wall clock may EXPIRE it, never
        # extend it — every sub-wait below stays monotonic.
        budget = spec.timeout_s + self.wait_slack_s
        deadline = time.monotonic() + budget
        wall_deadline = time.time() + budget
        last_heartbeat: float | None = None
        while True:
            remaining = deadline - time.monotonic()
            wall_expired = time.time() >= wall_deadline
            if remaining <= 0 or wall_expired:
                if remaining <= 0 and wall_expired:
                    expired = "both"
                elif remaining <= 0:
                    expired = "monotonic"
                else:
                    # Wall expiry with monotonic time to spare: the monotonic
                    # clock stood still — the suspend signature.
                    expired = "wall"
                self._note_lifecycle(
                    handle.task_id,
                    "timeout-fired",
                    expired_clock=expired,
                    timeout_s=spec.timeout_s,
                    mono_remaining_s=round(remaining, 3),
                )
                self._terminate(running)
                return SessionResult(
                    status="timeout",
                    session_id=self._session_id(running),
                    timeout_fired_at=time.time(),
                    timeout_expired_clock=expired,
                    stop_seen=running.result_event is not None,
                )
            # Hard-stop poll (#319), on both sides of the blocking wait below so
            # at most one leg sits between two checks. Return the verdict — never
            # raise, and never unlink the request file: the engine consumes it and
            # attributes the stop.
            if self._hard_stop_requested():
                self._note_lifecycle(handle.task_id, "stop-abort-fired")
                self._terminate(running)
                return SessionResult(
                    status="aborted",
                    session_id=self._session_id(running),
                    stop_seen=running.result_event is not None,
                )
            now = time.monotonic()
            if last_heartbeat is None or now - last_heartbeat >= HEARTBEAT_INTERVAL_S:
                last_heartbeat = now
                self._write_heartbeat(
                    handle.task_id, {"ts": time.time(), "remaining_s": round(remaining, 3)}
                )
                # Mid-session spec-status sampling (#276 M2): inert on this class,
                # real on the dev subclass via `_DevSynthesisMixin`.
                self._observe_tick(handle, spec)
            try:
                item = running.lines.get(timeout=min(remaining, self.poll_tick_s))
            except queue.Empty:
                continue
            if self._hard_stop_requested():
                self._note_lifecycle(handle.task_id, "stop-abort-fired")
                self._terminate(running)
                return SessionResult(
                    status="aborted",
                    session_id=self._session_id(running),
                    stop_seen=running.result_event is not None,
                )
            if item is _EOF:
                break
            self._consume(running, item if isinstance(item, str) else "")
        # stdout closed: the child is done talking, but "done talking" is not yet
        # "exited". Give it a brief reap window first — terminating a process that
        # was about to exit 0 on its own would turn a clean turn into a signalled
        # one — then fall back to `_terminate` for a child that closed stdout and
        # wedged.
        if running.proc is not None:
            try:
                running.proc.wait(timeout=self.exit_grace_s)
            except subprocess.TimeoutExpired:
                self._note_lifecycle(handle.task_id, "stdout-eof-without-exit")
        self._terminate(running)
        event = running.result_event
        session_id = self._session_id(running)
        usage = parse_usage(event)
        if session_id and usage is not None:
            self._usage[session_id] = usage
        failure = result_failure(event)
        if failure is not None:
            # Record only. The verdict below is unchanged by this: a frame that
            # ended the turn is a Stop whatever it says about how the turn went,
            # and `_final` still decides on the artifact.
            self._note_lifecycle(handle.task_id, "result-frame-reported-error", **failure)
        # The result frame ≙ Stop, its absence ≙ window death. Both are terminal
        # (the process is gone), so both vouch for a landed artifact; only the
        # fallback verdict differs. `_final` upgrades either to `completed` when
        # the read-back proves work, subject to the #261 proof-of-work gate that
        # `stop_seen` feeds.
        return self._final(
            handle,
            spec,
            "stalled" if event is not None else "crashed",
            session_id,
            str(self.logs_dir / f"{handle.task_id}.log"),
            stop_seen=event is not None,
        )

    def _consume(self, running: _Running, line: str) -> None:
        """Record the terminal control frame; ignore every other frame.

        Only a typed ``{"type": "result"}`` object counts. Model prose arrives as
        ``assistant``/``text`` frames and is deliberately dropped here — reading
        it would be the completion-on-LLM-prose path AGENTS.md forbids."""
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            return
        if isinstance(event, dict) and event.get("type") == "result":
            running.result_event = event

    @staticmethod
    def _session_id(running: _Running) -> str | None:
        event = running.result_event
        if not event:
            return None
        raw = event.get("session_id") or event.get("request_id")
        return str(raw) if raw else None

    def _terminate(self, running: _Running) -> None:
        """Idempotent teardown: stop the child, then close the stderr sink."""
        proc = running.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        for sink in running.sinks:
            try:
                sink.close()
            except OSError:
                pass
        running.sinks = []

    def kill(self, handle: SessionHandle) -> None:
        running = self._running.get(handle.task_id)
        if running is not None:
            self._terminate(running)

    def read_usage(self, result: SessionResult) -> TokenUsage | None:
        return self._usage.get(result.session_id) if result.session_id else None


class CursorCliHeadlessDevAdapter(_DevSynthesisMixin, CursorCliHeadlessAdapter):
    """Dev/review variant for the generic ``bmad-build-auto`` skill.

    That skill writes no ``result.json`` — its outcome is the terminal spec it
    leaves on disk, which :class:`_DevSynthesisMixin` locates and synthesizes via
    :mod:`devcontract`, the same machinery ``GenericDevAdapter`` and
    ``OpencodeDevAdapter`` use. Shared, never forked."""

    def __init__(self, *args: Any, paths: "ProjectPaths", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.paths = paths
        self._configure_dev_knobs()
        # `_configure_dev_knobs` arms the stall grace, the wake-nudge budget and
        # the #276 M4 contract nudge — all three assume a session that can be
        # injected into mid-turn. `-p` cannot be (see the module docstring), and
        # the contract nudge in particular sends through `send_text`, which this
        # family leaves raising. Disarm them rather than let the wait loop reach
        # a transport that cannot serve them.
        self._stall_grace_s = 0.0
        self._stall_nudges = 0
        self._contract_nudge_enabled = False

    def _probe_alive(self, handle: SessionHandle) -> bool | None:
        """Liveness for ``_post_kill_reconcile``. Never None: the live Popen
        handle pins the pid, so ``poll()`` is always answerable."""
        running = self._running.get(handle.task_id)
        if running is None or running.proc is None:
            return False  # never spawned: nothing we own is alive
        return running.proc.poll() is None


def validate_environment(project: Path) -> tuple[list[str], list[str]]:
    """Preflight notes/errors for ``bmad-loop validate``."""
    del project
    binary = shutil.which(BINARY)
    if binary is None:
        return [], [f"{BINARY} not found on PATH — install Cursor CLI headless support and re-run"]
    notes = [f"{BINARY} found ({binary})"]
    notes.append(
        "CURSOR_API_KEY is set"
        if os.environ.get("CURSOR_API_KEY")
        else "CURSOR_API_KEY unset — run `cursor-agent login` or export it"
    )
    return notes, []
