"""Supervised headless Cursor CLI provider (no tmux or Node SDK required)."""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import devcontract
from ..journal import LOGS_DIR
from ..model import TokenUsage
from .base import CodingCLIAdapter, SessionHandle, SessionResult, SessionSpec
from .profile import CLIProfile, HookSpec

if TYPE_CHECKING:
    from ..bmadconfig import ProjectPaths
    from ..policy import Policy, ResolvedAdapter
    from .adapter_kinds import AdapterKind

PROVIDER = "cursor-cli-headless"
BINARY = "cursor-agent"
DEFAULT_SKILL_TREE = ".cursor/skills"
RESULT_GRACE_S = 15.0
RESULT_POLL_S = 0.25
WAIT_SLACK_S = 30.0


def _profile() -> CLIProfile:
    return CLIProfile(
        name=PROVIDER,
        binary=BINARY,
        hooks=HookSpec("none", "", {}),
        skill_tree=DEFAULT_SKILL_TREE,
        usage_parser="none",
    )


def build_argv(*, prompt: str, cwd: Path, model: str = "", binary: str = BINARY) -> list[str]:
    argv = [
        binary,
        "-p",
        "--force",
        "--trust",
        "--output-format",
        "stream-json",
        "--workspace",
        str(cwd),
    ]
    if model.strip():
        argv.extend(["--model", model.strip()])
    return [*argv, prompt]


def parse_usage(event: dict[str, Any] | None) -> TokenUsage | None:
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
        output_tokens=integer("outputTokens") + integer("reasoningTokens"),
        cache_read_tokens=integer("cacheReadTokens"),
        cache_creation_tokens=integer("cacheWriteTokens"),
    )


def reconcile(
    event: dict[str, Any] | None, result_json: dict[str, Any] | None, *, timed_out: bool
) -> str:
    if timed_out:
        return "timeout"
    if event is not None:
        return "completed" if result_json is not None else "stalled"
    return "crashed"


@dataclass
class _Running:
    proc: subprocess.Popen[str] | None
    lines: queue.Queue[str | None]
    result_event: dict[str, Any] | None = None
    spawn_error: str | None = None


class CursorCliHeadlessAdapter(CodingCLIAdapter):
    name = PROVIDER
    injection = "launch-flag"
    observation = "stream"
    state = "local-json-tree"

    def __init__(
        self,
        run_dir: Path,
        policy: "Policy",
        *,
        model: str = "",
        binary: str = BINARY,
        paths: "ProjectPaths | None" = None,
        result_grace_s: float = RESULT_GRACE_S,
        wait_slack_s: float = WAIT_SLACK_S,
    ) -> None:
        self.run_dir, self.policy, self.model, self.binary = run_dir, policy, model, binary
        self.profile, self.paths = _profile(), paths
        self.result_grace_s, self.wait_slack_s = result_grace_s, wait_slack_s
        self.tasks_dir, self.logs_dir = run_dir / "tasks", run_dir / LOGS_DIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._running: dict[str, _Running] = {}
        self._usage: dict[str, TokenUsage] = {}

    def start_session(self, spec: SessionSpec) -> SessionHandle:
        task_dir = self.tasks_dir / spec.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "prompt.txt").write_text(spec.prompt + "\n", encoding="utf-8")
        (task_dir / "result.json").unlink(missing_ok=True)
        lines: queue.Queue[str | None] = queue.Queue()
        launched_ns = time.time_ns()
        try:
            proc = subprocess.Popen(
                build_argv(
                    prompt=spec.prompt,
                    cwd=spec.cwd,
                    model=spec.model or self.model,
                    binary=self.binary,
                ),
                cwd=spec.cwd,
                env={**os.environ, **spec.env},
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            running = _Running(proc, lines)
            threading.Thread(
                target=self._pump,
                args=(running, self.logs_dir / f"{spec.task_id}.log"),
                daemon=True,
            ).start()
            native_id = str(proc.pid)
        except (OSError, ValueError) as error:
            running = _Running(None, lines, spawn_error=str(error))
            lines.put(None)
            native_id = "spawn-failed"
        self._running[spec.task_id] = running
        return SessionHandle(spec.task_id, native_id, launched_ns)

    @staticmethod
    def _pump(running: _Running, log_path: Path) -> None:
        try:
            assert running.proc is not None and running.proc.stdout is not None
            with log_path.open("a", encoding="utf-8") as log:
                for line in running.proc.stdout:
                    log.write(line)
                    log.flush()
                    running.lines.put(line)
        finally:
            running.lines.put(None)

    def wait_for_completion(self, handle: SessionHandle, spec: SessionSpec) -> SessionResult:
        running = self._running.get(handle.task_id)
        if running is None or running.spawn_error is not None:
            return SessionResult(status="crashed")
        deadline = time.monotonic() + spec.timeout_s + self.wait_slack_s
        eof = False
        while not eof and time.monotonic() < deadline:
            try:
                line = running.lines.get(timeout=min(0.25, deadline - time.monotonic()))
            except queue.Empty:
                continue
            if line is None:
                eof = True
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                running.result_event = event
        timed_out = not eof and running.proc is not None and running.proc.poll() is None
        if timed_out:
            self._terminate(running)
        event = running.result_event
        session_id = str(event.get("session_id") or event.get("request_id")) if event else None
        usage = parse_usage(event)
        if session_id and usage is not None:
            self._usage[session_id] = usage
        result_json = None
        if event is not None:
            result_json = (
                self._await_synthesis(handle, spec)
                if spec.role in {"dev", "review"}
                else self._await_result(handle.task_id)
            )
        return SessionResult(
            reconcile(event, result_json, timed_out=timed_out), result_json, session_id
        )

    def _await_result(self, task_id: str) -> dict[str, Any] | None:
        path, deadline = (
            self.tasks_dir / task_id / "result.json",
            time.monotonic() + self.result_grace_s,
        )
        while True:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except (OSError, json.JSONDecodeError):
                pass
            if time.monotonic() >= deadline:
                return None
            time.sleep(RESULT_POLL_S)

    def _await_synthesis(self, handle: SessionHandle, spec: SessionSpec) -> dict[str, Any] | None:
        if self.paths is None:
            return None
        deadline = time.monotonic() + self.result_grace_s
        expected = Path(spec.expected_spec) if spec.expected_spec else None
        if expected is not None and not expected.is_absolute():
            expected = spec.cwd / expected
        while True:
            candidate = expected
            if candidate is None:
                dirs = [
                    self.paths.rebased(spec.cwd).implementation_artifacts,
                    self.paths.implementation_artifacts,
                ]
                for artifacts in dict.fromkeys(dirs):
                    candidate = devcontract.find_result_artifact(
                        artifacts, since_ns=handle.launched_ns
                    )
                    if candidate is not None:
                        break
            if candidate is not None and devcontract.is_result_artifact(
                candidate, since_ns=handle.launched_ns
            ):
                story_key = spec.env.get("BMAD_LOOP_STORY_KEY") or None
                dw_ids = [x for x in spec.env.get("BMAD_LOOP_DW_IDS", "").split(",") if x]
                return devcontract.synthesize_result(
                    candidate, story_key=story_key, dw_ids=dw_ids or None
                ).result_json
            if time.monotonic() >= deadline:
                return None
            time.sleep(RESULT_POLL_S)

    @staticmethod
    def _terminate(running: _Running) -> None:
        if running.proc is not None and running.proc.poll() is None:
            running.proc.terminate()
            try:
                running.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                running.proc.kill()

    def kill(self, handle: SessionHandle) -> None:
        running = self._running.pop(handle.task_id, None)
        if running is not None:
            self._terminate(running)

    def read_usage(self, result: SessionResult) -> TokenUsage | None:
        return self._usage.get(result.session_id) if result.session_id else None


def validate_environment(project: Path) -> tuple[list[str], list[str]]:
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


def cursor_cli_headless_kind() -> "AdapterKind":
    from ..bmadconfig import BmadConfigError, load_paths
    from .adapter_kinds import AdapterKind

    def build(
        *, run_dir: Path, policy: "Policy", cfg: "ResolvedAdapter", project: Path
    ) -> CursorCliHeadlessAdapter:
        try:
            paths = load_paths(project)
        except BmadConfigError:
            paths = None
        return CursorCliHeadlessAdapter(run_dir, policy, model=cfg.model, paths=paths)

    return AdapterKind(
        PROVIDER, build, _profile(), validate_environment, ("dev", "review", "triage")
    )
