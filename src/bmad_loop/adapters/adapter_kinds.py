"""Registry for coding providers whose transport is not a tmux profile."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import CodingCLIAdapter
    from .profile import CLIProfile


class ProvisionError(RuntimeError):
    """An optional provider runtime could not be installed."""


@dataclass(frozen=True)
class AdapterKind:
    name: str
    build: Callable[..., "CodingCLIAdapter"]
    profile: "CLIProfile"
    validate: Callable[[Path], tuple[list[str], list[str]]]
    roles: tuple[str, ...] | None = None
    provision: Callable[[], list[str]] | None = None


_KINDS: dict[str, AdapterKind] = {}
_BUILTINS_LOADED = False


def register_adapter_kind(kind: AdapterKind) -> None:
    _KINDS[kind.name] = kind


def _load_builtin_kinds() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from .cursor_cli_headless import cursor_cli_headless_kind

    register_adapter_kind(cursor_cli_headless_kind())
    _BUILTINS_LOADED = True


def get_adapter_kind(name: str) -> AdapterKind | None:
    _load_builtin_kinds()
    return _KINDS.get(name)


def is_adapter_kind(name: str) -> bool:
    return get_adapter_kind(name) is not None


def _reset_for_tests() -> None:
    global _BUILTINS_LOADED
    _KINDS.clear()
    _BUILTINS_LOADED = False
