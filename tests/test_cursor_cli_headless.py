"""Focused contracts for Cursor CLI's hookless stream-json provider."""

from __future__ import annotations

from conftest import install_bmad_config

from bmad_loop.adapters import cursor_cli_headless as headless
from bmad_loop.adapters.cursor_cli_headless import CursorCliHeadlessAdapter
from bmad_loop.policy import load


def _policy(project):
    path = project / ".bmad-loop" / "policy.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text('[adapter]\nname = "cursor-cli-headless"\n', encoding="utf-8")
    return load(path)


def test_headless_cursor_argv_forwards_skill_prompt_and_optional_model(tmp_path):
    argv = headless.build_argv(prompt="/bmad-loop-sweep", cwd=tmp_path, model="composer")
    assert argv == [
        "cursor-agent",
        "-p",
        "--force",
        "--trust",
        "--output-format",
        "stream-json",
        "--workspace",
        str(tmp_path),
        "--model",
        "composer",
        "/bmad-loop-sweep",
    ]


def test_headless_cursor_reconcile_and_usage():
    event = {
        "usage": {
            "inputTokens": 1,
            "outputTokens": 2,
            "reasoningTokens": 3,
            "cacheReadTokens": 4,
            "cacheWriteTokens": 5,
        }
    }
    assert headless.parse_usage(event).total == 15  # type: ignore[union-attr]
    assert headless.reconcile(event, {"status": "done"}, timed_out=False) == "completed"
    assert headless.reconcile(event, None, timed_out=False) == "stalled"
    assert headless.reconcile(None, {}, timed_out=False) == "crashed"


def test_headless_cursor_adapter_kind_bypasses_multiplexer(project, monkeypatch):
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
    assert isinstance(adapters["triage"], CursorCliHeadlessAdapter)
    assert adapters["dev"] is adapters["review"]
    assert adapters["triage"] is not adapters["dev"]
    assert adapters["dev"].profile.hookless


def test_init_headless_cursor_seeds_skill_tree_without_hook_relay(tmp_path):
    from bmad_loop import cli

    assert cli.main(["init", "--project", str(tmp_path), "--cli", "cursor-cli-headless"]) == 0
    assert (tmp_path / ".cursor" / "skills" / "bmad-loop-sweep" / "SKILL.md").is_file()
    assert not (tmp_path / ".bmad-loop" / "bmad_loop_hook.py").exists()
