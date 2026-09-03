"""Cursor SDK provider: the Node-sidecar transport, its registry wiring, and
the ``--provision`` runtime installer.

The family is selected the way every family now is — a packaged TOML profile
whose ``adapter`` field names a kind registered in ``adapters/registry.py`` — so
the seam-level tests here assert *dispatch* (``make_adapters`` builds the right
class and never resolves a multiplexer) rather than a private lookup table.

The transport tests drive a **scripted fake sidecar**: a Python script standing
in for ``node cursor-sidecar.mjs``, injected through the ``_sidecar_argv`` seam.
So they exercise the real spawn, the real stdout pump and the real verdict
logic, with no Node, no ``@cursor/sdk``, no API key and zero LLM tokens.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest
from conftest import install_bmad_config

from bmad_loop import cli
from bmad_loop import policy as policy_mod
from bmad_loop import runsetup
from bmad_loop.adapters import cursor_sdk
from bmad_loop.adapters import multiplexer as mux_mod
from bmad_loop.adapters.base import SessionSpec
from bmad_loop.adapters.cursor_sdk import (
    CursorSdkAdapter,
    CursorSdkDevAdapter,
    ProvisionError,
    parse_node_version,
    parse_usage,
)
from bmad_loop.adapters.profile import get_profile

CURSOR_POLICY = '[adapter]\nname = "cursor-sdk"\n'


def _write_policy(project: Path, text: str) -> None:
    d = project / ".bmad-loop"
    d.mkdir(parents=True, exist_ok=True)
    (d / "policy.toml").write_text(text, encoding="utf-8")


def _run_dir(project: Path) -> Path:
    return project / ".bmad-loop" / "runs" / "r"


def _adapter(project: Path, **kwargs) -> CursorSdkAdapter:
    return CursorSdkAdapter(
        _run_dir(project),
        policy_mod.load(None),
        get_profile("cursor-sdk"),
        **kwargs,
    )


def _spec(project: Path, **kwargs) -> SessionSpec:
    defaults = dict(
        task_id="t1",
        role="triage",
        prompt="/bmad-loop-sweep --all",
        cwd=project,
        timeout_s=30.0,
    )
    return SessionSpec(**{**defaults, **kwargs})


# --------------------------------------------------------------------------- #
# The scripted fake sidecar


def _fake_sidecar(tmp_path: Path, body: str) -> Path:
    """A Python stand-in for ``node cursor-sidecar.mjs``. ``body`` runs with
    ``emit(obj)`` (one NDJSON line on stdout) and ``opts`` (the parsed argv) in
    scope, so a test writes only what its case is about."""
    script = tmp_path / "fake_sidecar.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, sys
            argv = sys.argv[1:]
            opts = {argv[i].lstrip("-"): argv[i + 1] for i in range(0, len(argv) - 1, 2)}
            def emit(obj):
                sys.stdout.write(json.dumps(obj) + "\\n")
                sys.stdout.flush()
            """
        )
        + textwrap.dedent(body),
        encoding="utf-8",
    )
    return script


def _point_at(monkeypatch, adapter: CursorSdkAdapter, script: Path) -> None:
    monkeypatch.setattr(
        type(adapter),
        "_sidecar_argv",
        lambda self, spec, _script, prompt_file: [
            sys.executable,
            str(script),
            "--prompt-file",
            str(prompt_file),
            "--model",
            spec.model or cursor_sdk.DEFAULT_MODEL,
        ],
    )
    # The real `_ensure_sidecar_script` materializes the packaged .mjs into the
    # SDK home; harmless but pointless when argv ignores it, and it would write
    # into the developer's real ~/.bmad-loop.
    monkeypatch.setattr(type(adapter), "_ensure_sidecar_script", lambda self: script)


SENTINEL_OK = """
emit({"type": "assistant", "text": "working"})
emit({
    "type": "__sidecar_result__",
    "status": "finished",
    "agentId": "agent-1",
    "runId": "run-1",
    "usage": {"inputTokens": 11, "outputTokens": 22, "cacheReadTokens": 3, "cacheWriteTokens": 4},
})
"""

# The skill's own artifact, written by the session rather than planted by the
# test: `start_session` unlinks `result.json` at launch, which is exactly what
# makes a file found afterwards provably this session's.
WRITE_RESULT = """
import pathlib
path = pathlib.Path(opts["prompt-file"]).parent / "result.json"
path.write_text(json.dumps({"status": "done"}))
"""


# --------------------------------------------------------------------------- #
# Pure units


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("v22.13.0\n", (22, 13, 0)),
        ("22.14.1", (22, 14, 1)),
        ("v24.0.0 (some build)", (24, 0, 0)),
        ("not a version", None),
        ("", None),
    ],
)
def test_parse_node_version(text, expected):
    assert parse_node_version(text) == expected


def test_parse_usage_maps_the_sdk_keys_and_degrades_per_field():
    """Each column degrades independently: the SDK's usage shape has grown keys
    over time, so a missing or non-numeric one must cost that column, not the
    whole tally."""
    usage = parse_usage(
        {"usage": {"inputTokens": 5, "outputTokens": "oops", "cacheReadTokens": None}}
    )
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (5, 0)
    assert (usage.cache_read_tokens, usage.cache_creation_tokens) == (0, 0)


@pytest.mark.parametrize("sentinel", [None, {}, {"usage": None}, {"usage": "nope"}])
def test_parse_usage_is_none_without_a_usage_object(sentinel):
    """No usage object means "unknown", never a row of zeros — a fabricated zero
    tally is indistinguishable from a session that really spent nothing.

    ABLATION: replace the `isinstance(usage, dict)` guard with `usage or {}` and
    every row here reddens (each yields a zeroed TokenUsage instead of None)."""
    assert parse_usage(sentinel) is None


def test_packaged_profile_spells_the_skill_out_as_an_instruction():
    """A headless SDK run has no slash-command dispatcher, so the canonical
    "/skill args" prompt is rendered as prose. This is the profile's
    `prompt_template` doing it — the same layer every other CLI's rendering lives
    in — not adapter code."""
    text = get_profile("cursor-sdk").render_prompt("/bmad-loop-sweep --all")
    assert "Run the `bmad-loop-sweep` skill" in text
    assert "/bmad-loop-sweep" not in text
    assert "Invocation arguments: --all" in text


# --------------------------------------------------------------------------- #
# Dispatch through the registry seam


def test_make_adapters_builds_the_sdk_family_without_resolving_a_multiplexer(
    project, monkeypatch
):
    """The cursor-sdk profile routes to the Node-sidecar classes, and the shared
    multiplexer is never even resolved — the kind registers ``needs_mux=False``.
    Dev and review share the synthesizing variant; triage gets the plain one.

    ABLATION for the mux half: flip the cursor-sdk row in `_BUILTIN_ADAPTERS` to
    `needs_mux=True` and the monkeypatched `get_multiplexer` raises, reddening
    the test — a hookless family must not be able to fail a run on a host that
    has no terminal multiplexer."""

    def _boom():
        raise AssertionError("hookless cursor-sdk must not resolve a multiplexer")

    monkeypatch.setattr(mux_mod, "get_multiplexer", _boom)
    install_bmad_config(project)
    _write_policy(project.project, CURSOR_POLICY)

    adapters = runsetup.make_adapters(
        project.project, _run_dir(project.project), policy_mod.load(cli._policy_path(project.project))
    )

    assert isinstance(adapters["dev"], CursorSdkDevAdapter)
    assert adapters["dev"] is adapters["review"]
    assert isinstance(adapters["triage"], CursorSdkAdapter)
    assert not isinstance(adapters["triage"], CursorSdkDevAdapter)
    assert adapters["dev"].profile.adapter == "cursor-sdk"
    assert adapters["dev"].profile.hookless


def test_dev_variant_empties_the_nudge_budgets_it_cannot_spend(project, monkeypatch):
    """`_configure_dev_knobs` arms a stall grace and two nudge budgets for a
    transport that can talk to a live turn. A `@cursor/sdk` run ends when the
    sentinel lands and has no injection channel, so all three are emptied —
    otherwise the contract nudge would call `send_text`, which this family does
    not implement, and the base raises NotImplementedError out of the wait loop.

    ABLATION: delete the three assignments after `_configure_dev_knobs()` and the
    stall-grace row reddens (the policy default is non-zero)."""
    monkeypatch.setattr(mux_mod, "get_multiplexer", lambda: pytest.fail("no mux"))
    install_bmad_config(project)
    _write_policy(project.project, CURSOR_POLICY)
    dev = runsetup.make_adapters(
        project.project, _run_dir(project.project), policy_mod.load(cli._policy_path(project.project))
    )["dev"]
    assert dev._stall_grace_s == 0.0
    assert dev._stall_nudges == 0
    assert dev._stop_nudges == 0
    assert dev._contract_nudge_enabled is False


# --------------------------------------------------------------------------- #
# The transport, against the scripted fake sidecar


def test_a_finished_sentinel_plus_a_result_file_completes_and_tallies_usage(
    project, tmp_path, monkeypatch
):
    """The happy path end to end: spawn, drain the NDJSON, see the sentinel, read
    the skill's result.json back, and report the sentinel's usage object."""
    adapter = _adapter(project.project)
    _point_at(monkeypatch, adapter, _fake_sidecar(tmp_path, WRITE_RESULT + SENTINEL_OK))

    result = adapter.run(_spec(project.project))

    assert result.status == "completed"
    assert result.result_json == {"status": "done"}
    assert result.session_id == "run-1"
    assert result.stop_seen is True
    usage = adapter.read_usage(result)
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (11, 22)
    assert (usage.cache_read_tokens, usage.cache_creation_tokens) == (3, 4)


def test_the_rendered_prompt_reaches_the_sidecar_as_a_file(project, tmp_path, monkeypatch):
    """The prompt is handed over on disk, not on the command line: an argv token
    is size-bounded by the OS and would be mangled by quoting. The file holds the
    PROFILE-rendered text, so the sidecar never sees the raw "/skill args" form."""
    adapter = _adapter(project.project)
    _point_at(monkeypatch, adapter, _fake_sidecar(tmp_path, SENTINEL_OK))

    adapter.run(_spec(project.project))

    written = (adapter.tasks_dir / "t1" / "prompt.txt").read_text(encoding="utf-8")
    assert "Run the `bmad-loop-sweep` skill" in written
    assert "/bmad-loop-sweep" not in written


def test_a_finished_sentinel_without_an_artifact_is_a_stall_not_a_crash(
    project, tmp_path, monkeypatch
):
    """A clean sentinel is this transport's `Stop`: the turn genuinely ended, so
    a session that produced nothing stalled. Distinguishing that from `crashed`
    is what lets the engine spend a stall nudge rather than a dev attempt."""
    adapter = _adapter(project.project)
    _point_at(monkeypatch, adapter, _fake_sidecar(tmp_path, SENTINEL_OK))

    result = adapter.run(_spec(project.project))

    assert result.status == "stalled"
    assert result.result_json is None


@pytest.mark.parametrize(
    ("body", "label"),
    [
        ('emit({"type": "__sidecar_result__", "status": "error"})', "error-sentinel"),
        ('emit({"type": "assistant", "text": "no sentinel here"})', "no-sentinel"),
        ("", "silent-exit"),
    ],
)
def test_anything_short_of_a_finished_sentinel_and_no_artifact_is_a_crash(
    project, tmp_path, monkeypatch, body, label
):
    """Turn-end has exactly one signal — a sentinel saying ``finished``. A sidecar
    that errored, said nothing, or died mid-stream produced none, and with no
    artifact to read back the session crashed."""
    adapter = _adapter(project.project)
    _point_at(monkeypatch, adapter, _fake_sidecar(tmp_path, body))

    result = adapter.run(_spec(project.project))

    assert result.status == "crashed", label
    assert result.result_json is None, label
    assert result.stop_seen is False, label


def test_a_result_written_before_the_run_broke_still_counts(project, tmp_path, monkeypatch):
    """A skill that wrote its result and then lost the SDK run did the work. The
    read-back runs on the crash path too, exactly as it does on the tmux
    adapters' window-death path — and it is safe here because `start_session`
    unlinked the task-scoped path, so anything found afterwards is this
    session's."""
    adapter = _adapter(project.project)
    _point_at(
        monkeypatch,
        adapter,
        _fake_sidecar(
            tmp_path, WRITE_RESULT + 'emit({"type": "__sidecar_result__", "status": "error"})'
        ),
    )

    result = adapter.run(_spec(project.project))

    assert result.status == "completed"
    assert result.result_json == {"status": "done"}


def test_a_previous_attempts_result_is_unlinked_at_launch(project, tmp_path, monkeypatch):
    """`tasks/<task_id>/result.json` is task-scoped, so a leftover from an earlier
    attempt on the same id would read as this session's work. Cleared at launch,
    exactly as the tmux adapters do — and that unlink is the whole basis for the
    crash-path read-back above.

    ABLATION: drop the `unlink(missing_ok=True)` in `start_session` and this
    reddens — the stale artifact survives and the session reports `completed`."""
    adapter = _adapter(project.project)
    _point_at(
        monkeypatch,
        adapter,
        _fake_sidecar(tmp_path, 'emit({"type": "__sidecar_result__", "status": "error"})'),
    )
    task_dir = adapter.tasks_dir / "t1"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "result.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")

    result = adapter.run(_spec(project.project))

    assert result.status == "crashed"
    assert not (task_dir / "result.json").exists()


def test_a_sidecar_that_never_started_reports_no_proof_of_work(project, tmp_path, monkeypatch):
    """The #261 gate needs a session that died on arrival to read False (rendered
    nothing), not None (no such signal) — None is inert, so the gate would fail
    open in exactly the case it exists for, and the dev variant would then honor
    a spec some *other* run left in the shared artifacts dir.

    The DOA case is a spawn failure: the stdout pump never runs, so nothing else
    creates the log. Hence the `.touch()` before the spawn.

    ABLATION: drop that `.touch()` and the spawn-failure row reddens to None;
    drop the whole `_log_evidence` override and both the False and True rows do."""
    adapter = _adapter(project.project)
    assert adapter._log_evidence(cursor_sdk.SessionHandle("never", "0", 0)) is None

    monkeypatch.setattr(type(adapter), "_ensure_sidecar_script", lambda self: Path("unused"))
    monkeypatch.setattr(
        type(adapter),
        "_sidecar_argv",
        lambda self, spec, script, prompt_file: [str(project.project / "no-such-node")],
    )
    doa = adapter.start_session(_spec(project.project))
    assert adapter._log_evidence(doa) is False

    _point_at(monkeypatch, adapter, _fake_sidecar(tmp_path, 'emit({"pad": "x" * 512})'))
    chatty = adapter.start_session(_spec(project.project, task_id="t2"))
    adapter.wait_for_completion(chatty, _spec(project.project, task_id="t2"))
    assert adapter._log_evidence(chatty) is True


def test_a_sidecar_that_cannot_be_spawned_crashes_without_raising(project, monkeypatch):
    """A missing interpreter is an ordinary session outcome, not an orchestrator
    fault: the engine must get a `crashed` verdict it can journal, not an OSError
    unwinding the run."""
    adapter = _adapter(project.project)
    monkeypatch.setattr(
        type(adapter),
        "_sidecar_argv",
        lambda self, spec, script, prompt_file: [str(project.project / "no-such-node")],
    )
    monkeypatch.setattr(type(adapter), "_ensure_sidecar_script", lambda self: Path("unused"))

    result = adapter.run(_spec(project.project))

    assert result.status == "crashed"
    assert result.session_id is None


def test_stdout_is_tee_d_to_the_session_log_and_stderr_to_the_env_fault_log(
    project, tmp_path, monkeypatch
):
    """The two streams go to different files on purpose. `<task>.log` is the SDK
    event stream and therefore carries the model's own words; `<task>.sidecar.err`
    is the sidecar's own stderr, which the model cannot write to — so it is the
    only one env-fault patterns may be matched against, and it is what
    `ENV_FAULT_LOG_SUFFIX` names."""
    adapter = _adapter(project.project)
    _point_at(
        monkeypatch,
        adapter,
        _fake_sidecar(tmp_path, SENTINEL_OK + '\nsys.stderr.write("node: boom\\n")\n'),
    )

    adapter.run(_spec(project.project))

    assert '"working"' in (adapter.logs_dir / "t1.log").read_text(encoding="utf-8")
    assert adapter.ENV_FAULT_LOG_SUFFIX == ".sidecar.err"
    assert "node: boom" in (adapter.logs_dir / "t1.sidecar.err").read_text(encoding="utf-8")


def test_a_sidecar_that_never_finishes_times_out_and_is_torn_down(
    project, tmp_path, monkeypatch
):
    """The session clock is the backstop when no sentinel ever lands. The verdict
    carries which clock expired (#157), and the process is terminated rather than
    left running past the run."""
    adapter = _adapter(project.project)
    _point_at(
        monkeypatch,
        adapter,
        _fake_sidecar(tmp_path, 'import time\nemit({"type": "start"})\ntime.sleep(60)\n'),
    )
    adapter.poll_tick_s = 0.05

    result = adapter.wait_for_completion(
        adapter.start_session(_spec(project.project, timeout_s=0.4)),
        _spec(project.project, timeout_s=0.4),
    )

    assert result.status == "timeout"
    assert result.timeout_expired_clock in {"monotonic", "wall", "both"}
    assert adapter._sidecars["t1"].proc.poll() is not None


def test_a_hard_stop_request_aborts_the_session(project, tmp_path, monkeypatch):
    """#319: an operator's hard `bmad-loop stop` is honored mid-session. The
    verdict is returned, never raised — raising would skip `run()`'s finally-kill
    — and the request file is left for the engine to consume and attribute."""
    from bmad_loop import runs

    adapter = _adapter(project.project)
    _point_at(
        monkeypatch,
        adapter,
        _fake_sidecar(tmp_path, 'import time\nemit({"type": "start"})\ntime.sleep(60)\n'),
    )
    adapter.poll_tick_s = 0.05
    adapter.run_dir.mkdir(parents=True, exist_ok=True)
    (adapter.run_dir / "stop-request.json").write_text(
        json.dumps({"mode": "hard"}), encoding="utf-8"
    )

    spec = _spec(project.project)
    result = adapter.wait_for_completion(adapter.start_session(spec), spec)

    assert result.status == "aborted"
    assert runs.read_stop_request_mode(adapter.run_dir) == "hard"


def test_the_dev_variant_reports_sidecar_liveness(project, tmp_path, monkeypatch):
    """`_post_kill_reconcile` asks `_probe_alive` to settle liveness after the
    teardown. Unlike a tmux probe this one can always answer — the Popen handle
    pins the pid — so it is never the tristate `None`."""
    adapter = CursorSdkDevAdapter(
        _run_dir(project.project),
        policy_mod.load(None),
        get_profile("cursor-sdk"),
        paths=project,
    )
    _point_at(monkeypatch, adapter, _fake_sidecar(tmp_path, SENTINEL_OK))
    spec = _spec(project.project, role="dev")

    assert adapter._probe_alive(cursor_sdk.SessionHandle("t1", "0", 0)) is False
    handle = adapter.start_session(spec)
    adapter.wait_for_completion(handle, spec)
    adapter.kill(handle)
    assert adapter._probe_alive(handle) is False


# --------------------------------------------------------------------------- #
# Environment validation


def test_validate_environment_reports_all_three_preconditions(monkeypatch, tmp_path):
    """Node, the provisioned runtime and the API key are reported independently:
    an operator fixing one should not have to re-run to discover the next."""
    monkeypatch.setattr(cursor_sdk.envvars, "cursor_sdk_dir", lambda: str(tmp_path / "home"))
    monkeypatch.setattr(cursor_sdk.shutil, "which", lambda _b: None)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)

    notes, problems = cursor_sdk.validate_environment("node")

    assert notes == []
    assert any("not found on PATH" in p for p in problems)
    assert any("@cursor/sdk not found" in p and "--provision cursor-sdk" in p for p in problems)
    assert any("CURSOR_API_KEY is not set" in p for p in problems)


def test_validate_environment_is_green_when_everything_is_in_place(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / "node_modules" / "@cursor" / "sdk").mkdir(parents=True)
    (home / "node_modules" / "@cursor" / "sdk" / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cursor_sdk.envvars, "cursor_sdk_dir", lambda: str(home))
    monkeypatch.setattr(cursor_sdk.shutil, "which", lambda _b: "/usr/bin/node")
    monkeypatch.setattr(cursor_sdk, "_node_version", lambda _n: (22, 14, 0))
    monkeypatch.setenv("CURSOR_API_KEY", "k")

    notes, problems = cursor_sdk.validate_environment("node")

    assert problems == []
    assert any("node 22.14.0" in n for n in notes)


def test_validate_environment_refuses_a_node_below_the_floor(monkeypatch, tmp_path):
    """`@cursor/sdk` needs the Node 22 LTS line. An old Node is a distinct
    problem from a missing one, and says which floor it missed."""
    monkeypatch.setattr(cursor_sdk.envvars, "cursor_sdk_dir", lambda: str(tmp_path / "home"))
    monkeypatch.setattr(cursor_sdk.shutil, "which", lambda _b: "/usr/bin/node")
    monkeypatch.setattr(cursor_sdk, "_node_version", lambda _n: (20, 11, 0))

    _notes, problems = cursor_sdk.validate_environment("node")

    assert any("node 20.11.0 is below the 22.13 floor" in p for p in problems)


def test_validate_surfaces_the_cursor_sdk_preflight(project, tmp_path, monkeypatch, capsys):
    """`validate` reports the family's preconditions before a run can fail on
    them at launch, keyed on the adapter KIND — the same shape as the opencode
    family's httpx check.

    ABLATION: delete the `profile.adapter == CURSOR_SDK` block in `cmd_validate`
    and this reddens; nothing else emits an `adapter.cursor-sdk` finding."""
    install_bmad_config(project)
    _write_policy(project.project, CURSOR_POLICY)
    monkeypatch.setattr(cursor_sdk.envvars, "cursor_sdk_dir", lambda: str(tmp_path / "empty"))
    monkeypatch.setattr(cursor_sdk.shutil, "which", lambda _b: None)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)

    assert cli.main(["validate", "--project", str(project.project)]) == 1

    captured = capsys.readouterr()
    text = (captured.out + captured.err).lower()
    assert "cursor-sdk: 'node' not found on path" in text
    assert "cursor-sdk: @cursor/sdk not found" in text
    assert "--provision cursor-sdk" in text
    assert "cursor-sdk: cursor_api_key is not set" in text


# --------------------------------------------------------------------------- #
# Provisioning the @cursor/sdk runtime


def test_provision_is_idempotent_on_an_already_installed_runtime(monkeypatch, tmp_path):
    """Re-running `init --provision` must not re-shell to npm: provisioning is a
    network operation, and an already-present runtime is the answer."""
    home = tmp_path / "home"
    (home / "node_modules" / "@cursor" / "sdk").mkdir(parents=True)
    (home / "node_modules" / "@cursor" / "sdk" / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cursor_sdk.envvars, "cursor_sdk_dir", lambda: str(home))
    monkeypatch.setattr(
        cursor_sdk.shutil, "which", lambda _b: pytest.fail("npm must not be resolved")
    )

    assert cursor_sdk.provision_sdk() == [f"@cursor/sdk already provisioned at {home}"]


def test_provision_writes_a_package_json_and_shells_to_npm(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setattr(cursor_sdk.envvars, "cursor_sdk_dir", lambda: str(home))
    monkeypatch.setattr(cursor_sdk.shutil, "which", lambda _b: "/usr/bin/npm")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        pkg = home / "node_modules" / "@cursor" / "sdk"
        pkg.mkdir(parents=True)
        (pkg / "package.json").write_text("{}", encoding="utf-8")
        return cursor_sdk.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(cursor_sdk.subprocess, "run", fake_run)

    notes = cursor_sdk.provision_sdk()

    assert notes == [f"@cursor/sdk installed at {home}"]
    assert calls[0][0] == ["/usr/bin/npm", "install", "--no-audit", "--no-fund"]
    assert calls[0][1]["cwd"] == home
    manifest = json.loads((home / "package.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"] == {"@cursor/sdk": cursor_sdk.SDK_PIN}


def test_provision_without_npm_raises_a_provision_error(monkeypatch, tmp_path):
    monkeypatch.setattr(cursor_sdk.envvars, "cursor_sdk_dir", lambda: str(tmp_path / "home"))
    monkeypatch.setattr(cursor_sdk.shutil, "which", lambda _b: None)
    with pytest.raises(ProvisionError, match="npm not found"):
        cursor_sdk.provision_sdk()


def test_provision_reports_the_npm_failure_output(monkeypatch, tmp_path):
    """A failed install must name what npm said. The whole point of the fail-loud
    boundary is that the operator gets npm's own diagnosis, not "it didn't work"."""
    home = tmp_path / "home"
    monkeypatch.setattr(cursor_sdk.envvars, "cursor_sdk_dir", lambda: str(home))
    monkeypatch.setattr(cursor_sdk.shutil, "which", lambda _b: "/usr/bin/npm")
    monkeypatch.setattr(
        cursor_sdk.subprocess,
        "run",
        lambda argv, **kw: cursor_sdk.subprocess.CompletedProcess(argv, 1, "", "E404 not found"),
    )
    with pytest.raises(ProvisionError, match="E404 not found"):
        cursor_sdk.provision_sdk()


def test_init_provision_runs_the_kinds_installer(tmp_path, monkeypatch, capsys):
    """`bmad-loop init --provision cursor-sdk` resolves the kind through the
    registry and calls its provision thunk. The two flags are independent axes —
    `--cli` names profiles, `--provision` names adapter kinds — so the hook relay
    and the default claude profile are installed exactly as they always are."""
    monkeypatch.setattr(cursor_sdk, "provision_sdk", lambda: ["@cursor/sdk installed at /x"])

    assert cli.main(["init", "--project", str(tmp_path), "--provision", "cursor-sdk"]) == 0

    assert "@cursor/sdk installed at /x" in capsys.readouterr().out
    assert (tmp_path / ".bmad-loop" / "bmad_loop_hook.py").is_file()


def test_init_provision_of_a_kind_with_no_runtime_fails_naming_the_offer(tmp_path, capsys):
    """A kind that installs nothing is a usage error, not a silent no-op, and the
    message lists what IS provisionable — read off the live registry, never a
    hardcoded set."""
    assert cli.main(["init", "--project", str(tmp_path), "--provision", "generic"]) == 1
    out = capsys.readouterr().out
    assert "no runtime to install" in out
    assert "provisionable kinds: cursor-sdk" in out


def test_init_provision_of_an_unknown_kind_fails_loud(tmp_path, capsys):
    assert cli.main(["init", "--project", str(tmp_path), "--provision", "nosuch"]) == 1
    assert "unknown adapter kind 'nosuch'" in capsys.readouterr().out


def test_init_provision_failure_is_an_rc_not_a_traceback(tmp_path, monkeypatch, capsys):
    """The installer is a first-run command whose output an operator reads. A
    provisioner's declared failure becomes one FAIL line and rc 1."""

    def _boom():
        raise ProvisionError("npm install failed in /x: E404")

    monkeypatch.setattr(cursor_sdk, "provision_sdk", _boom)

    assert cli.main(["init", "--project", str(tmp_path), "--provision", "cursor-sdk"]) == 1
    assert "could not provision 'cursor-sdk': npm install failed in /x: E404" in (
        capsys.readouterr().out
    )
