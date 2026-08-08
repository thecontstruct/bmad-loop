"""Focused contract tests for the non-tmux Cursor SDK provider."""

from __future__ import annotations

from conftest import install_bmad_config

from bmad_loop.adapters import cursor_sdk
from bmad_loop.adapters.cursor_sdk import CursorSdkAdapter
from bmad_loop.policy import load


def _policy(project):
    path = project / ".bmad-loop" / "policy.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text('[adapter]\nname = "cursor-sdk"\n', encoding="utf-8")
    return load(path)


def test_cursor_sdk_render_prompt_uses_explicit_skill_instruction():
    text = cursor_sdk.render_prompt("/bmad-loop-sweep --all")
    assert "Run the `bmad-loop-sweep` skill" in text
    assert "/bmad-loop-sweep" not in text
    assert "--all" in text


def test_cursor_sdk_reconcile_requires_finished_sentinel_and_artifact():
    assert (
        cursor_sdk.reconcile({"status": "finished"}, {"status": "done"}, timed_out=False)
        == "completed"
    )
    assert cursor_sdk.reconcile({"status": "finished"}, None, timed_out=False) == "stalled"
    assert cursor_sdk.reconcile({"status": "error"}, {}, timed_out=False) == "crashed"
    assert cursor_sdk.reconcile({"status": "finished"}, {}, timed_out=True) == "timeout"


def test_cursor_sdk_adapter_kind_bypasses_multiplexer(project, monkeypatch):
    from bmad_loop import cli
    from bmad_loop.adapters import multiplexer

    install_bmad_config(project)
    monkeypatch.setattr(
        multiplexer, "get_multiplexer", lambda: (_ for _ in ()).throw(AssertionError())
    )
    adapters = cli._make_adapters(
        project.project,
        project.project / ".bmad-loop" / "runs" / "cursor",
        _policy(project.project),
    )
    assert isinstance(adapters["dev"], CursorSdkAdapter)
    assert adapters["dev"] is adapters["review"] is adapters["triage"]
    assert adapters["dev"].profile.hookless


def test_init_cursor_sdk_seeds_skills_without_a_hook_relay(tmp_path):
    from bmad_loop import cli

    assert cli.main(["init", "--project", str(tmp_path), "--cli", "cursor-sdk"]) == 0
    assert (tmp_path / ".cursor" / "skills" / "bmad-loop-sweep" / "SKILL.md").is_file()
    assert not (tmp_path / ".bmad-loop" / "bmad_loop_hook.py").exists()
