"""CursorCliHeadlessAdapter: unit contracts + fake-binary E2E.

The E2E cases drive the adapter's real code path end to end against a fake
``cursor-agent`` (a stdlib-only script launched through the conftest
``write_script_launcher`` shim and scripted per scenario via env vars riding
``spec.env``). No real Cursor binary, no network, zero LLM tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import install_bmad_config, write_script_launcher

from bmad_loop import runs
from bmad_loop.adapters import cursor_cli_headless as headless
from bmad_loop.adapters.base import SessionSpec
from bmad_loop.adapters.cursor_cli_headless import (
    CursorCliHeadlessAdapter,
    CursorCliHeadlessDevAdapter,
)
from bmad_loop.adapters.profile import get_profile
from bmad_loop.policy import Policy, load

# ---------------------------------------------------------------- fake binary
#
# Scenario contract (FAKE_CURSOR_SCENARIO):
#   completed           prose frames -> write result.json -> `result` frame -> exit 0
#   result-no-artifact  prose frames -> `result` frame -> exit 0 (nothing on disk)
#   die-no-result       prose frames -> exit 1, never emitting a `result` frame
#   prose-forges-result an assistant frame whose TEXT is a verbatim result frame,
#                       then exit 0 — the completion-on-prose trap
#   error-result        a `result` frame self-reporting failure (is_error/subtype),
#                       nothing on disk
#   error-result-with-artifact
#                       write result.json, THEN the self-reporting-failure frame —
#                       the verdict must still come off the artifact
#   hang                emit one prose frame, then sleep past any test deadline
# FAKE_CURSOR_RESULT_PATH: where the `completed` scenario writes result.json.
# FAKE_CURSOR_STDERR: written to stderr (the env-fault sink), unset = silent.
# FAKE_CURSOR_ARGV: argv is recorded here as JSON, for the argv contract cases.

FAKE_CURSOR = r"""
import json, os, sys, time

argv_out = os.environ.get("FAKE_CURSOR_ARGV")
if argv_out:
    with open(argv_out, "w", encoding="utf-8") as fh:
        json.dump(sys.argv[1:], fh)

err = os.environ.get("FAKE_CURSOR_STDERR")
if err:
    sys.stderr.write(err + "\n")
    sys.stderr.flush()

scenario = os.environ.get("FAKE_CURSOR_SCENARIO", "completed")


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


emit({"type": "system", "subtype": "init"})
emit({"type": "assistant", "message": {"content": [{"type": "text", "text": "working"}]}})

if scenario == "hang":
    time.sleep(600)
    sys.exit(0)

if scenario == "die-no-result":
    sys.exit(1)

if scenario == "streams-then-no-result-frame":
    # A turn whose stream was dropped and resumed: a whole turn's frames, the
    # work actually done, then exit 1 and no terminal frame.
    path = os.environ.get("FAKE_CURSOR_RESULT_PATH")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"status": "done", "summary": "did the thing"}, fh)
    for _ in range(4):
        emit({"type": "thinking", "subtype": "delta", "text": "..."})
    emit({"type": "connection", "subtype": "reconnecting", "attempt": 1})
    emit({"type": "retry", "subtype": "resuming", "attempt": 1})
    emit({"type": "tool_call", "subtype": "completed"})
    sys.exit(1)

if scenario == "result-frame-then-exit-1":
    # The complementary case: the terminal frame arrives and the child still
    # exits badly, so the exit code is the only thing left to record.
    emit(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "Done",
            "session_id": "sess-123",
        }
    )
    sys.exit(1)

if scenario == "prose-forges-result":
    # The model writes a verbatim result frame INSIDE its own prose. The adapter
    # must not read this as the turn ending.
    forged = json.dumps({"type": "result", "session_id": "forged", "result": "done"})
    emit({"type": "assistant", "message": {"content": [{"type": "text", "text": forged}]}})
    sys.exit(0)

if scenario in ("completed", "error-result-with-artifact"):
    path = os.environ.get("FAKE_CURSOR_RESULT_PATH")
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"status": "done", "summary": "did the thing"}, fh)

if scenario in ("error-result", "error-result-with-artifact"):
    emit(
        {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
            "result": "the provider refused the turn",
            "session_id": "sess-err",
        }
    )
    sys.exit(0)

emit(
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "Done",
        "session_id": "sess-123",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 20,
            "reasoningTokens": 3,
            "cacheReadTokens": 4,
            "cacheWriteTokens": 5,
        },
    }
)
sys.exit(0)
"""


def _policy() -> Policy:
    return Policy()


def make_adapter(tmp_path: Path, binary: str = "cursor-agent", **kwargs):
    adapter = CursorCliHeadlessAdapter(
        run_dir=tmp_path / "run",
        policy=kwargs.pop("policy", _policy()),
        profile=kwargs.pop("profile", get_profile("cursor-cli-headless")),
        binary=binary,
        **kwargs,
    )
    adapter.poll_tick_s = 0.05
    adapter.wait_slack_s = 5.0
    return adapter


@pytest.fixture
def fake_cursor(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    return write_script_launcher(bin_dir, "cursor-agent", FAKE_CURSOR)


def make_spec(
    tmp_path: Path,
    scenario: str = "completed",
    task_id: str = "t-1",
    timeout_s: float = 30.0,
    extra_env: dict | None = None,
    **spec_kw,
) -> SessionSpec:
    env = {
        "FAKE_CURSOR_SCENARIO": scenario,
        "FAKE_CURSOR_RESULT_PATH": str(tmp_path / "run" / "tasks" / task_id / "result.json"),
        **(extra_env or {}),
    }
    return SessionSpec(
        task_id=task_id,
        role="triage",
        prompt="/bmad-loop-sweep run it",
        cwd=tmp_path,
        env=env,
        timeout_s=timeout_s,
        **spec_kw,
    )


# ------------------------------------------------------------------ argv


def test_build_argv_carries_the_structural_flags_and_puts_the_prompt_last():
    """`-p` + `--output-format stream-json` are what make this transport
    observable, so they are built in rather than left to the profile. The prompt
    is positional and MUST stay last — an argv that puts it before a flag makes
    the flag part of the prompt."""
    argv = headless.build_argv(
        prompt="Use the bmad-loop-sweep skill now: run it",
        cwd=Path("/w"),
        model="composer",
        bypass=("--force", "--trust"),
    )
    assert argv == [
        "cursor-agent",
        "-p",
        "--force",
        "--trust",
        "--output-format",
        "stream-json",
        "--workspace",
        "/w",
        "--model",
        "composer",
        "Use the bmad-loop-sweep skill now: run it",
    ]


def test_build_argv_omits_the_model_flag_when_no_model_is_pinned():
    argv = headless.build_argv(prompt="p", cwd=Path("/w"))
    assert "--model" not in argv
    assert argv[-1] == "p"


def test_policy_extra_args_replace_the_profile_bypass_flags(tmp_path):
    """Same precedence as every other adapter: an explicit `extra_args` REPLACES
    the profile's bypass list rather than appending to it. `None` means unset."""
    profile = get_profile("cursor-cli-headless")
    assert profile.bypass_args == ("--force", "--trust")
    assert make_adapter(tmp_path).bypass_args == ("--force", "--trust")
    assert make_adapter(tmp_path, extra_args=("--yolo",)).bypass_args == ("--yolo",)
    # () is "no flags", a different state from None ("fall back to the profile").
    assert make_adapter(tmp_path, extra_args=()).bypass_args == ()


def test_launch_argv_renders_the_prompt_through_the_profile_template(tmp_path, fake_cursor):
    """The engine hands every adapter the canonical `/skill args` string; the
    adapter owes it to the profile template. Passing `spec.prompt` straight
    through would ship a bare slash command this transport does not expand.

    Ablation: drop the `render_prompt` call in `start_session` and the recorded
    argv tail becomes `/bmad-loop-sweep run it`, failing this row."""
    argv_path = tmp_path / "argv.json"
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    spec = make_spec(tmp_path, extra_env={"FAKE_CURSOR_ARGV": str(argv_path)})
    adapter.run(spec)
    argv = json.loads(argv_path.read_text(encoding="utf-8"))
    assert argv[-1] == "Use the bmad-loop-sweep skill now: run it"
    assert argv[:2] == ["-p", "--force"]
    assert "--workspace" in argv


# ----------------------------------------------------------------- usage


def test_parse_usage_does_not_add_reasoning_tokens_to_output():
    """Cursor documents `reasoningTokens` as a SUBSET of `outputTokens`
    (`@cursor/sdk` usage-types.d.ts; `totalTokens` excludes it), so adding the
    two double-counts output. This diverges from `tokens.py`'s copilot and gemini
    parsers, which do fold their vendors' reasoning counts in under the same key
    spelling — a deliberate, vendor-specific divergence, not an oversight.

    INVERSE ablation: the gate is the ABSENCE of `+ integer("reasoningTokens")`.
    Restore that addend in `parse_usage` and this row reddens on
    `output_tokens == 5` and `total == 15`."""
    usage = headless.parse_usage(
        {
            "usage": {
                "inputTokens": 1,
                "outputTokens": 2,
                "reasoningTokens": 3,
                "cacheReadTokens": 4,
                "cacheWriteTokens": 5,
            }
        }
    )
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (1, 2)
    assert (usage.cache_read_tokens, usage.cache_creation_tokens) == (4, 5)
    assert usage.total == 12


def test_parse_usage_maps_a_real_result_frames_usage_object():
    """The bytes of a real `cursor-agent` 2026.08.04 terminal frame. Its four
    counts sum to the `totalTokens` that vendor's SDK reports for the same
    object, which is the arithmetic proof reasoning is not a separate addend."""
    usage = headless.parse_usage(
        {
            "usage": {
                "inputTokens": 23314,
                "outputTokens": 275,
                "cacheReadTokens": 41984,
                "cacheWriteTokens": 0,
            }
        }
    )
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens) == (23314, 275)
    assert usage.total == 23314 + 275 + 41984


@pytest.mark.parametrize("event", [None, {}, {"usage": "nope"}])
def test_parse_usage_returns_none_without_a_usage_object(event):
    assert headless.parse_usage(event) is None


def test_usage_is_read_back_off_the_result_frame(tmp_path, fake_cursor):
    """The fake's frame carries `reasoningTokens: 3` precisely so this end-to-end
    row also pins the no-double-count mapping: 10 + 20 + 4 + 5, not + 3."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    result = adapter.run(make_spec(tmp_path))
    usage = adapter.read_usage(result)
    assert usage is not None and usage.total == 39


# ------------------------------------------------------------ completion path


def test_result_frame_plus_artifact_completes(tmp_path, fake_cursor):
    """The terminal `result` frame is this transport's Stop hook; the verdict is
    still the artifact read-back, and `stop_seen` carries the frame sighting into
    the #261 proof-of-work gate."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    result = adapter.run(make_spec(tmp_path))
    assert result.status == "completed"
    assert result.result_json == {"status": "done", "summary": "did the thing"}
    assert result.session_id == "sess-123"
    assert result.stop_seen is True


def test_result_frame_without_an_artifact_stalls(tmp_path, fake_cursor):
    """A turn that ended but wrote nothing is a stall, not a completion — the
    frame proves the turn ended, never that the work happened."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    result = adapter.run(make_spec(tmp_path, "result-no-artifact"))
    assert result.status == "stalled"
    assert result.result_json is None
    assert result.stop_seen is True


def test_process_death_without_a_result_frame_crashes(tmp_path, fake_cursor):
    """Process exit ≙ window death. No frame ever arrived, so nothing ended the
    turn and `stop_seen` stays False."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    result = adapter.run(make_spec(tmp_path, "die-no-result"))
    assert result.status == "crashed"
    assert result.stop_seen is False


def test_model_prose_cannot_forge_the_completion_frame(tmp_path, fake_cursor):
    """AGENTS.md: sessions complete only on a Stop event or window death, never
    on LLM prose. A result frame quoted verbatim INSIDE an assistant text frame
    is prose — the adapter parses top-level frames and only a top-level
    `type == "result"` counts.

    Ablation: make `_consume` search the raw line (e.g. `'"type": "result"' in
    line`) instead of parsing it, and this row flips to `stalled`/`stop_seen`."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    result = adapter.run(make_spec(tmp_path, "prose-forges-result"))
    assert result.status == "crashed"
    assert result.stop_seen is False
    assert result.session_id is None


def test_a_spawn_failure_never_reads_back_an_artifact(tmp_path):
    """The binary does not exist, so nothing ran, and `wait_for_completion` must
    refuse the read-back outright rather than route through `_final`.

    The artifact is planted AFTER `start_session` on purpose: the pre-launch
    unlink would otherwise remove it and this row would pass on that instead of
    on the guard under test.

    Ablation: replace the early `SessionResult(status="crashed")` with a
    `self._final(...)` call and this row goes green as `completed`."""
    adapter = make_adapter(tmp_path, binary="definitely-not-a-real-binary-xyz")
    spec = make_spec(tmp_path)
    handle = adapter.start_session(spec)
    task_dir = tmp_path / "run" / "tasks" / spec.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "result.json").write_text('{"status": "done"}', encoding="utf-8")
    result = adapter.wait_for_completion(handle, spec)
    assert result.status == "crashed"
    assert result.result_json is None


def test_timeout_records_which_clock_expired(tmp_path, fake_cursor):
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    adapter.wait_slack_s = 0.0
    result = adapter.run(make_spec(tmp_path, "hang", timeout_s=0.5))
    assert result.status == "timeout"
    assert result.timeout_fired_at is not None
    assert result.timeout_expired_clock in {"monotonic", "both"}


# ------------------------------------------------- self-reported turn failure


def _lifecycle(adapter, task_id: str) -> list[dict]:
    path = adapter.tasks_dir / task_id / "session-lifecycle.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _failure_crumbs(adapter, task_id: str) -> list[dict]:
    return [
        crumb
        for crumb in _lifecycle(adapter, task_id)
        if crumb["event"] == "result-frame-reported-error"
    ]


@pytest.mark.parametrize(
    "event",
    [
        None,
        {},
        "not a dict",
        {"type": "result", "subtype": "success", "is_error": False, "result": "Done"},
        {"type": "result", "is_error": False},
        {"type": "result", "subtype": "  success  "},
        {"type": "result", "subtype": 7},
    ],
)
def test_result_failure_stays_silent_on_a_frame_that_reports_no_failure(event):
    """A success frame, a frame with neither field, and a non-dict must all read
    as "no failure reported" — an absent signal is not a failure signal.

    Ablation: drop the `not is_error and (not subtype or subtype in
    SUCCESS_SUBTYPES)` early-out and every row here reddens with a dict."""
    assert headless.result_failure(event) is None


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            {"type": "result", "subtype": "error_during_execution", "is_error": True},
            {"subtype": "error_during_execution", "is_error": True},
        ),
        # `is_error` alone, no subtype: still a failure report.
        ({"type": "result", "is_error": True}, {"subtype": "", "is_error": True}),
        # An unrecognised subtype with `is_error` absent: recorded, because a
        # crumb costs a log line and a missed one costs a diagnosis.
        (
            {"type": "result", "subtype": "error_max_turns"},
            {"subtype": "error_max_turns", "is_error": False},
        ),
    ],
)
def test_result_failure_reports_the_frames_own_failure_fields(event, expected):
    assert headless.result_failure(event) == expected


def test_result_failure_carries_the_short_result_text_and_caps_it():
    """The frame's own one-line outcome is the whole payload — `.log` holds the
    model's stream and this breadcrumb file must not become a second copy of it,
    so the text is capped rather than pasted whole."""
    assert headless.result_failure({"is_error": True, "result": "  boom  "})["result"] == "boom"
    long = headless.result_failure({"is_error": True, "result": "x" * 5000})
    assert long is not None
    assert len(long["result"]) == headless.RESULT_DETAIL_MAX_CHARS
    # An empty or non-string `result` contributes no key at all.
    assert "result" not in headless.result_failure({"is_error": True, "result": "   "})
    assert "result" not in headless.result_failure({"is_error": True, "result": 7})


def test_a_self_reported_failure_frame_drops_a_lifecycle_breadcrumb(tmp_path, fake_cursor):
    """The CLI hands us a free, authoritative failure signal on the terminal
    frame; discarding it made a failed turn indistinguishable from a successful
    one in the run directory. It is recorded, never acted on.

    Ablation: delete the `_note_lifecycle(..., "result-frame-reported-error")`
    call from `wait_for_completion` and this row finds no crumb."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    spec = make_spec(tmp_path, "error-result")
    adapter.run(spec)
    crumbs = _failure_crumbs(adapter, spec.task_id)
    assert len(crumbs) == 1
    assert crumbs[0]["subtype"] == "error_during_execution"
    assert crumbs[0]["is_error"] is True
    assert crumbs[0]["result"] == "the provider refused the turn"


def test_a_successful_frame_drops_no_failure_breadcrumb(tmp_path, fake_cursor):
    """The gate is "the frame reported a failure", not "a frame arrived": a
    clean turn must leave the lifecycle file free of failure noise, or the crumb
    is worthless as a diagnosis.

    Ablation: make the `if failure is not None` arm in `wait_for_completion`
    unconditional and this row reddens on the emptiness assertion."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    spec = make_spec(tmp_path)
    result = adapter.run(spec)
    assert result.status == "completed"
    assert _failure_crumbs(adapter, spec.task_id) == []


def test_a_self_reported_failure_does_not_change_the_verdict(tmp_path, fake_cursor):
    """AGENTS.md: the verdict is deterministic and artifact-derived. A frame
    that ended the turn is this transport's Stop whatever it says about how the
    turn went, so `stop_seen` stays True and the fallback stays `stalled` —
    `crashed` would claim the process died without ending its turn, which is
    false here and would route the session into the #489 crash-diagnosis path.

    Ablation: make the fallback depend on the self-report (e.g. `"crashed" if
    failure else "stalled"`) and the first block reddens; make the failure frame
    veto the read-back and the second block reddens."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    failed = adapter.run(make_spec(tmp_path, "error-result", task_id="t-err"))
    assert failed.status == "stalled"
    assert failed.stop_seen is True
    assert failed.session_id == "sess-err"

    # Same self-reported failure, but the artifact landed: the read-back still
    # wins, because the frame is a report and the file is the evidence.
    spec = make_spec(tmp_path, "error-result-with-artifact", task_id="t-err-2")
    landed = adapter.run(spec)
    assert landed.status == "completed"
    assert landed.result_json == {"status": "done", "summary": "did the thing"}
    assert _failure_crumbs(adapter, spec.task_id)


# ------------------------------------------------- a turn that never ends itself


def _crumbs(adapter, task_id: str, event: str) -> list[dict]:
    return [crumb for crumb in _lifecycle(adapter, task_id) if crumb["event"] == event]


def test_a_stream_that_ends_without_its_terminal_frame_is_recorded(tmp_path, fake_cursor):
    """On a path that drops the agent stream, cursor-agent streams a whole turn,
    does the work, then exits 1 emitting no terminal frame. Nothing downstream fails —
    the artifact read-back still passes the session — so without this crumb a
    transport whose completion signal has disappeared entirely reads as a clean
    green run, and the 0-token columns are the only hint.

    Ablation: delete the `stream-ended-without-result-frame` `_note_lifecycle`
    call and this finds no crumb."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    spec = make_spec(tmp_path, "streams-then-no-result-frame")
    result = adapter.run(spec)

    crumbs = _crumbs(adapter, spec.task_id, "stream-ended-without-result-frame")
    assert len(crumbs) == 1
    assert crumbs[0]["exit_code"] == 1
    # A whole turn's frames arrived; only the terminal one is missing.
    assert crumbs[0]["frames_seen"] > 1

    # The verdict is untouched by the crumb: the artifact still carries it.
    assert result.status == "completed"
    assert result.result_json == {"status": "done", "summary": "did the thing"}


def test_frames_seen_separates_a_broken_stream_from_a_dead_child(tmp_path, fake_cursor):
    """A dropped stream and a child that dies having said nothing both end with
    no terminal frame, so the crumb alone cannot tell them apart — `frames_seen`
    is the whole reason it is worth recording. A dead child streams almost
    nothing; a dropped stream streams a full turn.

    Ablation: stop incrementing `running.frames_seen` in `_consume` and the two
    counts collapse to 0, reddening the inequality."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))

    broken = make_spec(tmp_path, "streams-then-no-result-frame", task_id="t-broken")
    adapter.run(broken)
    dead = make_spec(tmp_path, "die-no-result", task_id="t-dead")
    adapter.run(dead)

    broken_crumb = _crumbs(adapter, broken.task_id, "stream-ended-without-result-frame")[0]
    dead_crumb = _crumbs(adapter, dead.task_id, "stream-ended-without-result-frame")[0]
    assert broken_crumb["frames_seen"] > dead_crumb["frames_seen"]
    # Both exit 1: the code alone is not the discriminator, which is why the
    # count is carried at all.
    assert broken_crumb["exit_code"] == dead_crumb["exit_code"] == 1


def test_a_terminal_frame_with_a_bad_exit_code_records_the_code(tmp_path, fake_cursor):
    """A frame arrived, so the turn ended properly, but the child still exited
    non-zero. Nothing derives a verdict from an exit code — this crumb is the one
    place it is recorded, and it must not be swallowed by the no-frame path.

    Ablation: drop the `elif exit_code not in (0, None)` arm and this finds no
    crumb."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    spec = make_spec(tmp_path, "result-frame-then-exit-1")
    adapter.run(spec)

    crumbs = _crumbs(adapter, spec.task_id, "child-exit-nonzero")
    assert len(crumbs) == 1
    assert crumbs[0]["exit_code"] == 1
    # The two arms are exclusive: a frame did arrive, so the no-frame crumb must
    # stay absent or "no terminal frame" stops meaning anything.
    assert _crumbs(adapter, spec.task_id, "stream-ended-without-result-frame") == []


def test_a_clean_turn_records_neither_crumb(tmp_path, fake_cursor):
    """Both crumbs are diagnoses, so a healthy run must leave the lifecycle file
    free of them, or neither is worth reading.

    Ablation: make either `_note_lifecycle` call unconditional and this reddens."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    spec = make_spec(tmp_path)
    result = adapter.run(spec)
    assert result.status == "completed"
    assert _crumbs(adapter, spec.task_id, "stream-ended-without-result-frame") == []
    assert _crumbs(adapter, spec.task_id, "child-exit-nonzero") == []


def test_a_missing_terminal_frame_still_withholds_stop_seen(tmp_path, fake_cursor):
    """The one change here that could turn a silently-broken transport into a
    silently-passing one would be flipping `stop_seen` true to paper over a
    dropped frame: it feeds the #261 proof-of-work gate, which has teeth precisely
    because no Stop was observed. Recording the ending must not license it.

    Ablation: set `stop_seen=True` unconditionally in the `_final` call and this
    reddens."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    spec = make_spec(tmp_path, "streams-then-no-result-frame")
    result = adapter.run(spec)
    assert result.stop_seen is False
    # And no session id, because that too rides the frame — which is why usage
    # cannot be recovered on this path.
    assert result.session_id is None
    assert adapter.read_usage(result) is None


# --------------------------------------------------------------- hard stop


def _lodge_stop_request(adapter, mode: str) -> Path:
    """Lodge a stop request on this run's control-file channel, as
    ``bmad-loop stop`` does."""
    adapter.run_dir.mkdir(parents=True, exist_ok=True)
    path = adapter.run_dir / runs.STOP_REQUEST_FILE
    path.write_text(
        json.dumps({"requested_at": "2026-08-22T00:00:00", "mode": mode}), encoding="utf-8"
    )
    return path


def test_wait_aborts_on_a_hard_stop_request(tmp_path, fake_cursor):
    """#319: a hard stop pending on the channel ends the wait with the
    non-completion `aborted` verdict instead of running out the clock, and the
    request file is left on disk for the engine to consume and attribute.

    Ablation: delete the `_hard_stop_requested()` arms from `wait_for_completion`
    and this row hangs until the timeout, landing `timeout`."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    request = _lodge_stop_request(adapter, "hard")
    result = adapter.run(make_spec(tmp_path, "hang", timeout_s=60.0))
    assert result.status == "aborted"
    assert request.is_file()  # never unlinked by the adapter


def test_wait_ignores_a_graceful_stop_request(tmp_path, fake_cursor):
    """Only `mode: "hard"` aborts mid-session; a graceful stop is honored at the
    next item boundary by the engine, not here.

    Ablation: widen `_hard_stop_requested` past its `== "hard"` comparison and
    this row reddens with an `aborted` verdict."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    _lodge_stop_request(adapter, "graceful")
    result = adapter.run(make_spec(tmp_path))
    assert result.status == "completed"


# ------------------------------------------------------------------- logs


def test_stdout_and_stderr_land_in_separate_sinks(tmp_path, fake_cursor):
    """Two audiences: `<task>.log` is the model's own stream (the transcript),
    `<task>.err` is the CLI's diagnostics. Env-fault classification scans `.err`
    precisely because the model cannot write to it — a pattern matched against a
    log carrying model output is unsound.

    Ablation: point stderr back at DEVNULL and the `.err` assertion fails; merge
    the two sinks and the `.log` "not in" assertion fails."""
    adapter = make_adapter(tmp_path, binary=str(fake_cursor))
    spec = make_spec(tmp_path, extra_env={"FAKE_CURSOR_STDERR": "auth token expired"})
    adapter.run(spec)
    logs = tmp_path / "run" / "logs"
    transcript = (logs / f"{spec.task_id}.log").read_text(encoding="utf-8")
    diagnostics = (logs / f"{spec.task_id}.err").read_text(encoding="utf-8")
    assert '"type": "result"' in transcript
    assert "auth token expired" in diagnostics
    assert "auth token expired" not in transcript


def test_env_fault_scan_targets_the_stderr_sink():
    """The suffix is part of the shipped patterns' safety contract, so it is
    pinned here rather than left to whichever file a fixture happens to write."""
    assert CursorCliHeadlessAdapter.ENV_FAULT_LOG_SUFFIX == ".err"


def test_shipped_profile_seeds_no_env_fault_patterns():
    """No real outage line has been captured from this CLI yet, and a guessed
    pattern is worse than none."""
    assert get_profile("cursor-cli-headless").env_fault_patterns == ()


# ------------------------------------------------------------------ profile


def test_profile_is_hookless_and_names_its_adapter_kind():
    """A hookless profile MUST name its kind: leaving `adapter` at the `generic`
    default would dispatch to the tmux adapter, which completes on a Stop hook
    this profile never registers."""
    profile = get_profile("cursor-cli-headless")
    assert profile.hookless is True
    assert profile.adapter == "cursor-cli-headless"
    assert profile.binary == "cursor-agent"
    assert profile.skill_tree == ".cursor/skills"
    assert profile.usage_parser == "none"


def test_profile_prompt_template_names_the_skill_in_prose():
    """`-p` is one shot and promises no slash-command expansion, so the template
    names the skill in prose the way the opencode profile does."""
    rendered = get_profile("cursor-cli-headless").render_prompt("/bmad-loop-sweep run it")
    assert rendered == "Use the bmad-loop-sweep skill now: run it"


# ------------------------------------------------------- registry + bootstrap


def test_dev_and_plain_variants_are_distinct_classes():
    """The `dev`/`plain` split is the pipeline concept `runsetup.make_adapters`
    branches on: a bmad-build-auto session writes no result.json, so it needs the
    `_DevSynthesisMixin` variant that synthesizes from the spec on disk.

    Ablation: register the plain class for both and this row fails — which is
    exactly the state the branch shipped in before the synthesis mixin was
    reused instead of forked."""
    from bmad_loop.adapters.registry import get_adapter_kind

    builder = get_adapter_kind("cursor-cli-headless").load()
    assert builder.plain is CursorCliHeadlessAdapter
    assert builder.dev is CursorCliHeadlessDevAdapter
    assert builder.plain is not builder.dev


def test_dev_variant_reuses_the_shared_synthesis_mixin():
    """Shared, never forked: the bmad-build-auto read-back (expected_spec pinning,
    stories mode, the #224 missing-marker fallback, the #261 proof-of-work gate)
    lives in `_DevSynthesisMixin` and must not be reimplemented per transport."""
    from bmad_loop.adapters.generic import _DevSynthesisMixin, _ResultFileMixin

    assert issubclass(CursorCliHeadlessDevAdapter, _DevSynthesisMixin)
    assert issubclass(CursorCliHeadlessAdapter, _ResultFileMixin)
    assert CursorCliHeadlessDevAdapter._READBACK_NEEDS_PROOF_OF_WORK is True


def test_dev_variant_disarms_the_nudges_it_cannot_deliver(tmp_path, project):
    """`-p` cannot be injected into mid-turn, so `send_text` stays unimplemented.
    The contract nudge (#276 M4) sends through it, so it is disarmed rather than
    left to raise out of the wait loop.

    Ablation: drop the three disarming lines from `__init__` and the nudge budget
    assertions fail, leaving a wait loop that would call a raising transport."""
    adapter = CursorCliHeadlessDevAdapter(
        tmp_path / "run",
        _policy(),
        get_profile("cursor-cli-headless"),
        paths=project,
    )
    assert adapter._stall_nudges == 0
    assert adapter._stall_grace_s == 0.0
    assert adapter._stop_nudges == 0
    assert adapter._contract_nudge_enabled is False
    with pytest.raises(NotImplementedError):
        adapter.send_text(None, "nudge")  # type: ignore[arg-type]


def test_adapter_kind_bypasses_the_multiplexer(project, monkeypatch):
    """`needs_mux=False`: the bootstrap must never even resolve a multiplexer for
    this family. The monkeypatch is a trap, not an absence check — it raises if
    the seam is touched."""
    from bmad_loop import cli
    from bmad_loop.adapters import multiplexer

    install_bmad_config(project)
    monkeypatch.setattr(
        multiplexer, "get_multiplexer", lambda: (_ for _ in ()).throw(AssertionError())
    )
    policy_path = project.project / ".bmad-loop" / "policy.toml"
    policy_path.parent.mkdir(exist_ok=True)
    policy_path.write_text('[adapter]\nname = "cursor-cli-headless"\n', encoding="utf-8")
    adapters = cli._make_adapters(
        project.project,
        project.project / ".bmad-loop" / "runs" / "cursor",
        load(policy_path),
    )
    assert isinstance(adapters["triage"], CursorCliHeadlessAdapter)
    assert isinstance(adapters["dev"], CursorCliHeadlessDevAdapter)
    assert adapters["dev"] is adapters["review"]
    assert adapters["triage"] is not adapters["dev"]
    assert adapters["dev"].profile.hookless


# ---------------------------------------------------------------------- init


def test_init_seeds_the_skill_tree_and_writes_no_hook_relay(tmp_path):
    """A hookless-only init has nothing to relay, so the script is not written.

    Ablation: drop the `any(not profile.hookless ...)` guard in `install_into`
    and the relay reappears, failing the second assertion."""
    from bmad_loop import cli

    assert cli.main(["init", "--project", str(tmp_path), "--cli", "cursor-cli-headless"]) == 0
    assert (tmp_path / ".cursor" / "skills" / "bmad-loop-sweep" / "SKILL.md").is_file()
    assert not (tmp_path / ".bmad-loop" / "bmad_loop_hook.py").exists()


def test_init_still_writes_the_relay_when_a_hook_driven_cli_is_selected(tmp_path):
    """The guard is "no hook-driven profile selected", not "cursor was selected":
    mixing a hookless CLI with a hook-driven one must still install the relay."""
    from bmad_loop import cli

    rc = cli.main(
        ["init", "--project", str(tmp_path), "--cli", "cursor-cli-headless", "--cli", "claude"]
    )
    assert rc == 0
    assert (tmp_path / ".bmad-loop" / "bmad_loop_hook.py").is_file()


# ------------------------------------------------------------------ preview


def test_dry_run_preview_is_rendered_from_the_adapters_own_argv_builder(project):
    """The preview must not drift from what a run executes, so it is built by
    `build_argv` rather than hand-spelled in cli.py."""
    from bmad_loop import cli

    install_bmad_config(project)
    policy_path = project.project / ".bmad-loop" / "policy.toml"
    policy_path.parent.mkdir(exist_ok=True)
    policy_path.write_text(
        '[adapter]\nname = "cursor-cli-headless"\nmodel = "composer"\n', encoding="utf-8"
    )
    rendered = cli._render_invocation(
        load(policy_path), project.project, "dev", "/bmad-loop-sweep run it"
    )
    assert rendered == (
        "cursor-agent -p --force --trust --output-format stream-json "
        '--workspace <worktree> --model composer "Use the bmad-loop-sweep skill now: run it"'
    )
