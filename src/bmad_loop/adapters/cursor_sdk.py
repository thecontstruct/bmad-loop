"""Headless Cursor SDK adapter, supervised through the bundled Node sidecar.

The Cursor SDK has an in-band terminal result instead of bmad-loop hook files,
so this is intentionally a sibling of the tmux adapters.  It accepts the
current :class:`SessionSpec` contract, including an expected spec path for
safe dev/review result synthesis.
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

PROVIDER = "cursor-sdk"
DEFAULT_MODEL = "composer-2.5"
DEFAULT_SKILL_TREE = ".cursor/skills"
MIN_NODE = (22, 13)
SDK_PIN = "^1.0.23"
RESULT_GRACE_S = 15.0
RESULT_POLL_S = 0.25
WAIT_SLACK_S = 30.0
SENTINEL_TYPE = "__sidecar_result__"
_NODE_VERSION_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")


def _node_bin() -> str:
    return os.environ.get("BMAD_LOOP_NODE_BIN", "node")


def _sdk_home() -> Path:
    override = os.environ.get("BMAD_LOOP_CURSOR_SDK_DIR")
    return Path(override).expanduser() if override else Path.home() / ".bmad-loop" / PROVIDER


def _profile() -> CLIProfile:
    return CLIProfile(
        name=PROVIDER,
        binary=_node_bin(),
        hooks=HookSpec("none", "", {}),
        skill_tree=DEFAULT_SKILL_TREE,
        usage_parser="none",
    )


def parse_node_version(text: str) -> tuple[int, int, int] | None:
    match = _NODE_VERSION_RE.search(text.strip())
    return tuple(map(int, match.groups())) if match else None  # type: ignore[return-value]


def parse_usage(sentinel: dict[str, Any] | None) -> TokenUsage | None:
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


def render_prompt(prompt: str) -> str:
    """Use an explicit instruction; Cursor headless discovery is not slash based."""
    if not prompt.startswith("/"):
        return prompt
    skill, _, arguments = prompt[1:].partition(" ")
    lines = [
        "You are running fully headless in the bmad-loop automation harness.",
        f"Run the `{skill}` skill available under `{DEFAULT_SKILL_TREE}` and follow it to completion.",
    ]
    if arguments.strip():
        lines.append(f"Invocation arguments: {arguments.strip()}")
    lines.append("BMAD_LOOP_MODE, BMAD_LOOP_RUN_DIR, and BMAD_LOOP_TASK_ID are already set.")
    return "\n".join(lines)


def reconcile(
    sentinel: dict[str, Any] | None, result_json: dict[str, Any] | None, *, timed_out: bool
) -> str:
    if timed_out:
        return "timeout"
    if sentinel is not None and sentinel.get("status") == "finished":
        return "completed" if result_json is not None else "stalled"
    return "crashed"


@dataclass
class _Running:
    proc: subprocess.Popen[str] | None
    lines: queue.Queue[str | None]
    sentinel: dict[str, Any] | None = None
    spawn_error: str | None = None


class CursorSdkAdapter(CodingCLIAdapter):
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
        paths: "ProjectPaths | None" = None,
        sdk_home: Path | None = None,
        result_grace_s: float = RESULT_GRACE_S,
        wait_slack_s: float = WAIT_SLACK_S,
    ) -> None:
        self.run_dir, self.policy, self.model = run_dir, policy, model or DEFAULT_MODEL
        self.profile = _profile()
        self.paths, self.sdk_home = paths, sdk_home or _sdk_home()
        self.result_grace_s, self.wait_slack_s = result_grace_s, wait_slack_s
        self.tasks_dir, self.logs_dir = run_dir / "tasks", run_dir / LOGS_DIR
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._running: dict[str, _Running] = {}
        self._usage: dict[str, TokenUsage] = {}

    def _ensure_sidecar(self) -> Path:
        target = self.sdk_home / "cursor-sidecar.mjs"
        want = resources.files("bmad_loop.data").joinpath("cursor-sidecar.mjs").read_bytes()
        self.sdk_home.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_bytes() != want:
            target.write_bytes(want)
        return target

    def start_session(self, spec: SessionSpec) -> SessionHandle:
        task_dir = self.tasks_dir / spec.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        rendered = render_prompt(spec.prompt)
        prompt_file = task_dir / "sidecar-prompt.txt"
        prompt_file.write_text(rendered, encoding="utf-8")
        (task_dir / "prompt.txt").write_text(rendered + "\n", encoding="utf-8")
        (task_dir / "result.json").unlink(missing_ok=True)
        launched_ns = time.time_ns()
        lines: queue.Queue[str | None] = queue.Queue()
        try:
            argv = [
                _node_bin(),
                str(self._ensure_sidecar()),
                "--cwd",
                str(spec.cwd),
                "--model",
                spec.model or self.model,
                "--prompt-file",
                str(prompt_file),
                "--timeout-ms",
                str(int(spec.timeout_s * 1000)),
            ]
            proc = subprocess.Popen(
                argv,
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
            if isinstance(event, dict) and event.get("type") == SENTINEL_TYPE:
                running.sentinel = event
        timed_out = not eof and running.proc is not None and running.proc.poll() is None
        if timed_out:
            self._terminate(running)
        sentinel = running.sentinel
        session_id = str(sentinel.get("runId") or sentinel.get("agentId")) if sentinel else None
        usage = parse_usage(sentinel)
        if session_id and usage is not None:
            self._usage[session_id] = usage
        result_json = None
        if sentinel is not None and sentinel.get("status") == "finished":
            result_json = (
                self._await_synthesis(handle, spec)
                if spec.role in {"dev", "review"}
                else self._await_result(handle.task_id)
            )
        return SessionResult(
            reconcile(sentinel, result_json, timed_out=timed_out), result_json, session_id
        )

    def _await_result(self, task_id: str) -> dict[str, Any] | None:
        deadline = time.monotonic() + self.result_grace_s
        path = self.tasks_dir / task_id / "result.json"
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


def _node_version(node: str) -> tuple[int, int, int] | None:
    try:
        return parse_node_version(
            subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10).stdout
        )
    except (OSError, subprocess.SubprocessError):
        return None


def validate_environment(project: Path) -> tuple[list[str], list[str]]:
    del project
    notes, problems = [], []
    node = shutil.which(_node_bin())
    if node is None:
        problems.append(
            f"node not found on PATH — {PROVIDER} needs Node >= {MIN_NODE[0]}.{MIN_NODE[1]}"
        )
    elif (version := _node_version(node)) is None or version < MIN_NODE:
        problems.append(f"node must be >= {MIN_NODE[0]}.{MIN_NODE[1]} for {PROVIDER}")
    else:
        notes.append(f"node {'.'.join(map(str, version))} found ({node})")
    package = _sdk_home() / "node_modules" / "@cursor" / "sdk" / "package.json"
    if package.is_file():
        notes.append(f"@cursor/sdk present at {_sdk_home()}")
    else:
        problems.append(
            f"@cursor/sdk not found under {_sdk_home()}/node_modules — run `bmad-loop init --provision {PROVIDER}`"
        )
    if os.environ.get("CURSOR_API_KEY"):
        notes.append("CURSOR_API_KEY is set")
    else:
        problems.append("CURSOR_API_KEY is not set")
    return notes, problems


def provision_sdk() -> list[str]:
    from .adapter_kinds import ProvisionError

    home = _sdk_home()
    package = home / "node_modules" / "@cursor" / "sdk" / "package.json"
    if package.is_file():
        return [f"@cursor/sdk already provisioned at {home}"]
    npm = shutil.which("npm")
    if npm is None:
        raise ProvisionError(
            f"npm not found; install Node >= {MIN_NODE[0]}.{MIN_NODE[1]} then run `npm install` in {home}"
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
        completed = subprocess.run(
            [npm, "install", "--no-audit", "--no-fund"],
            cwd=home,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ProvisionError(f"could not install @cursor/sdk in {home}: {error}") from error
    if completed.returncode or not package.is_file():
        raise ProvisionError(
            f"npm install failed in {home}: {(completed.stderr or completed.stdout).strip()[-1000:]}"
        )
    return [f"@cursor/sdk installed at {home}"]


def cursor_sdk_kind() -> "AdapterKind":
    from ..bmadconfig import BmadConfigError, load_paths
    from .adapter_kinds import AdapterKind

    def build(
        *, run_dir: Path, policy: "Policy", cfg: "ResolvedAdapter", project: Path
    ) -> CursorSdkAdapter:
        try:
            paths = load_paths(project)
        except BmadConfigError:
            paths = None
        return CursorSdkAdapter(run_dir, policy, model=cfg.model, paths=paths)

    return AdapterKind(
        PROVIDER,
        build,
        _profile(),
        validate_environment,
        ("dev", "review", "triage"),
        provision_sdk,
    )
