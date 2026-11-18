"""Model of sprint-status.yaml — the single source of workflow truth.

The dev primitive `bmad-build-auto` deliberately does not touch sprint-status
("the orchestrator's business"), so the orchestrator is the single writer via
:func:`advance` — idempotent, never-regress, epic-lift. The orchestrator
otherwise only re-reads this file to pick the next story and verify what a
session claims.

Never-regress has exactly one kind of exception, and it is opt-in per call:
``advance(..., allow_regression=True)`` may move a row backward only through a
pair allowlisted in :data:`~bmad_loop.statemachine.BOARD_REGRESSIONS` (today
``done -> awaiting-operator``, a review pass demoting a finished story to a park,
DW-383). Any other regressing pair raises
:class:`~bmad_loop.statemachine.IllegalTransition` and writes nothing, so legal
board moves still live only in ``statemachine.py``.

Concurrency (#286/#469): being the sole writer is not on its own mutual
exclusion — a second orchestrator process (another `bmad-loop run`, a sweep, the
TUI) runs the same sole writer, and :func:`advance` is a read-modify-write of the
whole board, so two of them would both read, both edit, and let the last atomic
write win. :func:`advance` therefore serializes itself cross-process on the
board's state-root sidecar lock, and holds it across every read that decides the
PUBLISHED BYTES as well as the write itself. That invariant is deliberately
narrower than "every read": one advisory pre-lock probe may answer a
read-dependent no-op — an absent row, or a row already at or past target —
without acquiring at all (#736), because such a call publishes nothing and so has
no bytes for the hold to protect. Readers stay lock-free: the publish is an
atomic replace, so a reader sees either the old board entire or the new one.

Refusal is loud (#842): the writer is a line edit, not a YAML round-trip, so some
shapes the parser reads it will not rewrite. An existing row :func:`advance` has
decided to move and cannot rewrite raises :class:`SprintStatusWriteRefused`
(board, row, current, target, a stable reason) and publishes nothing, rather than
echoing the unchanged status a caller would read as "the session never got there".
An absent row, a missing board, and a row already at or past target still answer
by return value. The writer finds a row by the key :func:`load` reads, not by its
text (DW-515): ``str()`` of the constructed key, with epic numbers as ints, so
``epic-01`` is epic 1. Two distinct keys that read as one
(``"2001-01-04"`` and the date ``2001-01-04``, or ``epic-1`` and ``epic-01``) are
refused as ``key-ambiguous``, the epic lift included.

Retro action items (DW-388): ``bmad-retrospective`` appends the items a retro
commits to a top-level ``action_items:`` list, each with a stable ``id`` and a
``ref`` back to the retro document. :func:`load_action_items` reads that list on
its own, independently of :func:`load`, so ``development_status`` parsing — and
the ``RETRO_ITEM_RE``-keyed rows it recognizes — is untouched by it. The sweep is
its one consumer: it files each unseen, not-``done`` item into the deferred-work
ledger. Nothing here writes the list.
"""

from __future__ import annotations

import bisect
import re
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from .platform_util import atomic_write_bytes, file_lock
from .statemachine import check_board_regression

EPIC_RE = re.compile(r"^epic-(\d+)$")
RETRO_RE = re.compile(r"^epic-(\d+)-retrospective$")
RETRO_ITEM_RE = re.compile(r"^epic-(\d+)-retro-item-(\d+)-(.+)$")
# The story number may carry a single lowercase split suffix (2-6a / 2-6b —
# the shape BMAD produces when an oversized story is split, see issue #144).
STORY_RE = re.compile(r"^(\d+)-(\d+)([a-z]?)-(.+)$")
SHORT_REF_RE = re.compile(r"^(\d+)[-.](\d+)([a-z]?)$")  # short story ref: 3-1, 3.1, 3-1a
BARE_NUM_RE = re.compile(r"^(\d+)([a-z]?)$")  # a lone story number, needs --epic

# Lifecycle order, earliest -> latest. `advance` never moves a story backward
# through this sequence (matches sync-sprint-status's "never regress") — save the
# single allowlisted opt-in regression `done -> awaiting-operator` (DW-383,
# `statemachine.BOARD_REGRESSIONS`, `allow_regression=True`) — and it is the only
# ordering any caller may use: a token absent from it cannot be ordered at all, so
# every consumer treats "unknown" conservatively rather than guessing.
# `awaiting-operator` sits immediately before `done`: parking is the last stop on
# the way to finished, so confirming a parked story is a legal forward advance
# through the sole writer, while `done` regresses back into it only through that
# explicit opt-in (a review demotion under `on_review_demotion = "park"`).
STATUS_ORDER = (
    "backlog",
    "ready-for-dev",
    "in-progress",
    "review",
    "awaiting-operator",
    "done",
)
LEGACY_STORY_STATUSES = {"drafted": "ready-for-dev"}
# Statuses a story may be PICKED UP from. `awaiting-operator` is deliberately
# absent: the story's agent-doable work is already committed, so re-driving it
# would redo finished work while the human's external actions stay outstanding.
ACTIONABLE_STATUSES = {"backlog", "ready-for-dev"}


class SprintStatusError(Exception):
    pass


# Why the writer declined an existing row, as a stable token a caller may journal
# or branch on. Only :func:`_set_mapping_value`'s refusals and the one row
# :func:`_advance_locked` cannot find an entry for are named; message text is for
# operators and may change.
RefusalReason = Literal[
    "key-not-plain",  # the key is quoted, flow-style, or not where its line says
    "value-not-scalar",  # a nested mapping or sequence
    "value-is-alias",  # `*anchor`: the value's text is authored on another row
    "value-is-anchored",  # `&anchor value`: aliases elsewhere resolve to this text
    "mapping-is-flow",  # the row sits in a `{...}` flow mapping, not a block one
    "multiline-value",  # runs past the key line in a shape the collapse cannot read
    "unreadable-value",  # one line, but the text around the value span is not gap + comment
    "row-not-in-mapping",  # the reader resolves the key only through a `<<` merge
    "key-ambiguous",  # two or more distinct keys read as this one key (DW-515)
]

_REFUSAL_TEXT: dict[RefusalReason, str] = {
    "key-not-plain": "its key is not a plain `key:` at the start of its line",
    "value-not-scalar": "its value is a nested mapping or sequence",
    "value-is-alias": "its value is an alias to a node authored elsewhere",
    "value-is-anchored": "its value carries an anchor that other rows may alias",
    "mapping-is-flow": "it sits in a flow-style `{...}` mapping, not a block one",
    "multiline-value": "its value runs past the key line in a shape the writer cannot collapse",
    "unreadable-value": "the text after its key is not a value the writer can read whole",
    "row-not-in-mapping": "development_status reaches it only through a `<<` merge",
    "key-ambiguous": "two or more rows of development_status read as this one key",
}


class SprintStatusWriteRefused(SprintStatusError):
    """An existing story row had to move, and the line-edit writer would not rewrite it
    (or its epic lift hit ``key-ambiguous``; ``story_key`` is then ``epic-N``).

    Raised by :func:`advance` (and :func:`advanced_bytes`) only after the locked,
    authoritative read found the row and the call's own decision was to write it
    — below ``target``, or an allowlisted regression — so it never stands for an
    absent row, a missing board, or a row already at or past ``target``. Nothing
    has been written when it is raised: not the row, not an epic lift, not
    ``last_updated``.

    ``path`` is the board, or None when the refusal came from
    :func:`advanced_bytes`, which works on bytes and never names a file.
    ``current`` is the row's status as :func:`story_status` reads it; ``reason``
    is a stable :data:`RefusalReason` token.

    The epic lift raises it too, for ``key-ambiguous`` only (DW-515): two distinct
    keys reading as the story's epic (``epic-1`` and ``epic-01``) leave no row the
    lift could edit that is certainly the one :func:`load` reports. ``story_key`` is
    then the epic's canonical key ``epic-N`` and ``current`` its status as
    :func:`load` reads it; the story row is not written either. Every other epic
    refusal stays best effort and raises nothing."""

    path: Path | None
    story_key: str
    current: str
    target: str
    reason: RefusalReason

    def __init__(
        self,
        path: Path | None,
        story_key: str,
        current: str,
        target: str,
        reason: RefusalReason,
    ) -> None:
        self.path = path
        self.story_key = story_key
        self.current = current
        self.target = target
        self.reason = reason
        board = str(path) if path is not None else "the sprint-status board"
        if reason == "key-ambiguous":
            repair = (
                f"keep exactly one row reading as `{story_key}` (delete or rename the"
                " others), then retry."
            )
        else:
            repair = (
                f"rewrite the row as a plain one-line `{story_key}: <status>` entry, then retry."
            )
        super().__init__(
            f"sprint status row {story_key!r} in {board} is {current!r} and could not be"
            f" rewritten to {target!r}: {_REFUSAL_TEXT[reason]} ({reason}). The board was"
            f" left unchanged; {repair}"
        )


@dataclass(frozen=True)
class Story:
    key: str
    epic: int
    num: int
    slug: str
    status: str
    suffix: str = ""  # split-story letter ("a" in 2-6a), "" for a whole story


@dataclass(frozen=True)
class RetroItem:
    """A retrospective action item tracked in sprint-status under the
    RETRO ACTION ITEMS section: ``epic-{epic}-retro-item-{num}-{slug}``.

    Recognized so they no longer fall into ``unknown_keys``; the orchestrator
    does not yet drive them as work (see roadmap: retro-item automation).
    """

    key: str
    epic: int
    num: int
    slug: str
    status: str


@dataclass(frozen=True)
class SprintStatus:
    path: Path
    epics: dict[int, str]
    stories: tuple[Story, ...]
    retros: dict[int, str]
    retro_items: tuple[RetroItem, ...]
    unknown_keys: tuple[str, ...]


def load(path: Path) -> SprintStatus:
    if not path.is_file():
        raise SprintStatusError(f"sprint status file not found: {path}")
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SprintStatusError(f"sprint status is not valid YAML: {path}: {e}") from e
    if not isinstance(doc, dict):
        raise SprintStatusError(f"sprint status has no top-level mapping: {path}")
    dev = doc.get("development_status")
    if not isinstance(dev, dict):
        raise SprintStatusError(f"sprint status missing development_status map: {path}")

    epics: dict[int, str] = {}
    stories: list[Story] = []
    retros: dict[int, str] = {}
    retro_items: list[RetroItem] = []
    unknown: list[str] = []
    for key, raw_status in dev.items():
        key = str(key)
        status = str(raw_status).strip()
        if m := RETRO_ITEM_RE.match(key):
            retro_items.append(
                RetroItem(
                    key=key,
                    epic=int(m.group(1)),
                    num=int(m.group(2)),
                    slug=m.group(3),
                    status=status,
                )
            )
        elif m := RETRO_RE.match(key):
            retros[int(m.group(1))] = status
        elif m := EPIC_RE.match(key):
            epics[int(m.group(1))] = status
        elif m := STORY_RE.match(key):
            status = LEGACY_STORY_STATUSES.get(status, status)
            stories.append(
                Story(
                    key=key,
                    epic=int(m.group(1)),
                    num=int(m.group(2)),
                    slug=m.group(4),
                    status=status,
                    suffix=m.group(3),
                )
            )
        else:
            unknown.append(key)

    return SprintStatus(
        path=path,
        epics=epics,
        stories=tuple(stories),
        retros=retros,
        retro_items=tuple(retro_items),
        unknown_keys=tuple(unknown),
    )


@dataclass(frozen=True)
class ActionItem:
    """One id-keyed entry of sprint-status's top-level ``action_items:`` list,
    as ``bmad-retrospective`` appends it (DW-388).

    ``id`` is stripped and never empty — an entry without one is not an
    ActionItem at all (see :func:`load_action_items`). ``action`` is None when the
    YAML value is not a string, and kept verbatim otherwise (it may be blank; the
    consumer decides what a blank action means). ``owner`` and ``ref`` are None
    unless they are non-blank strings, and are stripped. ``status`` is the stripped
    string form of the value, ``""`` when absent; it is NOT validated, because an
    unknown token must not make an item disappear."""

    id: str
    epic: int | None
    action: str | None
    owner: str | None
    status: str
    ref: str | None


class ActionItemsMalformed(SprintStatusError):
    """The board parsed, but its ``action_items`` value is not a list.

    A subclass so a caller can tell the one shape fault that is about the retro
    list itself from a board that could not be read at all, without matching on
    message text."""


def _opt_str(value: object) -> str | None:
    """A non-blank string, stripped — or None for anything else."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _opt_epic(value: object) -> int | None:
    """The epic number as an int, from an int or an all-decimal string — None
    otherwise. ``bool`` is refused explicitly: it is an ``int`` subclass, and
    ``epic: true`` names no epic. ``isdecimal`` rather than ``isdigit``: the latter
    accepts ``"²"``/``"①"``, which ``int()`` then refuses with a bare ValueError."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdecimal():
        return int(value.strip())
    return None


def load_action_items(path: Path) -> tuple[ActionItem, ...] | None:
    """The board's id-keyed ``action_items``, in file order (DW-388).

    Returns None when the board path does not exist — absence is not a fault,
    and the caller has nothing to ingest — and ``()`` when the board has no
    ``action_items`` key, or the key holds no value (``action_items:`` with
    nothing under it, which the retrospective's own writer treats as an empty
    list too).

    Raises :class:`SprintStatusError` when the path is not a regular file, or
    the file cannot be read or decoded,
    is not valid YAML, or has no top-level mapping, and
    :class:`ActionItemsMalformed` (a subclass) when ``action_items`` holds
    something other than a list. Deliberately NOT :func:`load`: that reader also
    requires a ``development_status`` map, and a retro list must stay readable —
    and ``load`` byte-for-byte unchanged — whatever the rest of the board holds.

    Entries that are not mappings, or carry no non-empty string ``id``, are
    skipped: a legacy item without a stable id has no identity a consumer could
    dedupe on, so it cannot be ingested safely at all.

    Existence is probed with ``stat()`` rather than ``is_file()``, for the reason
    the sweep's triage-cache read gives (DW-224): ``is_file()`` swallows a
    metadata fault on 3.14 and re-raises it bare on 3.11-3.13, so a permission
    fault would read as "no board" on one runtime and escape untyped on another.
    Here it is a :class:`SprintStatusError` on all of them. A path that exists
    but is not a regular file (a directory, say) is a fault too, not absence: a
    degrade that read it as "no board" would be indistinguishable from one."""
    try:
        mode = path.stat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as e:
        raise SprintStatusError(f"sprint status could not be read: {path}: {e}") from e
    if not stat.S_ISREG(mode):
        raise SprintStatusError(f"sprint status is not a regular file: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise SprintStatusError(f"sprint status could not be read: {path}: {e}") from e
    try:
        doc = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise SprintStatusError(f"sprint status is not valid YAML: {path}: {e}") from e
    if not isinstance(doc, dict):
        raise SprintStatusError(f"sprint status has no top-level mapping: {path}")
    listed = doc.get("action_items")
    if listed is None:
        return ()
    if not isinstance(listed, list):
        raise ActionItemsMalformed(
            f"sprint status action_items is not a list ({type(listed).__name__}): {path}"
        )
    items: list[ActionItem] = []
    for entry in listed:
        if not isinstance(entry, dict):
            continue
        item_id = _opt_str(entry.get("id"))
        if item_id is None:
            continue
        action = entry.get("action")
        raw_status = entry.get("status")
        items.append(
            ActionItem(
                id=item_id,
                epic=_opt_epic(entry.get("epic")),
                action=action if isinstance(action, str) else None,
                owner=_opt_str(entry.get("owner")),
                status="" if raw_status is None else str(raw_status).strip(),
                ref=_opt_str(entry.get("ref")),
            )
        )
    return tuple(items)


def next_actionable(
    ss: SprintStatus, skip: set[str] | None = None, *, epic: int | None = None
) -> Story | None:
    """First story in file order whose status allows starting work. When
    ``epic`` is given, only stories of that epic are considered — the caller
    uses this to exhaust the current epic before advancing to another."""
    skip = skip or set()
    for story in ss.stories:
        if story.key in skip:
            continue
        if epic is not None and story.epic != epic:
            continue
        if story.status in ACTIONABLE_STATUSES:
            return story
    return None


def story_status(path: Path, key: str) -> str | None:
    """Fresh re-read of one story's status, for post-session verification."""
    ss = load(path)
    for story in ss.stories:
        if story.key == key:
            return story.status
    return None


_Scope = Literal["root", "development_status"]

# A node's leading properties: each an anchor (`&a`) or tag (`!t`, `!!str`,
# `!<uri>`) token, a run of non-space characters ended by spaces/tabs, a line
# break, or the end of the text. The scan never crosses a line break, so a
# property-only value (`key: !!str`) cannot claim the next row's `&anchor`. No
# plain scalar may open with `&` or `!`, so a value starting with either starts
# with a property.
_NODE_PROPERTIES_RE = re.compile(r"(?:[&!]\S*(?:[ \t]+|(?=[\r\n])|\Z))*")


@dataclass(frozen=True)
class _RowSpan:
    """Where one mapping entry's key and scalar value sit in the source.

    Line numbers index the caller's ``lines``; columns count characters from the
    start of their line. ``value_*`` is the value node's own extent, so a value
    that runs past its first line says so here whatever its indentation. It
    starts at the node's first property when it has one (a tag), and a null
    value's extent is empty, sitting right after the key's colon."""

    key_line: int
    key_col: int
    value_first_line: int
    value_start_col: int
    value_last_line: int
    value_end_col: int
    style: str | None  # PyYAML's ScalarNode.style: None for plain
    value: str
    # the key node's parsed text; equal to the source for the plain keys the editor accepts
    key_text: str


def _locate_row(lines: list[str], key: str, scope: _Scope) -> _RowSpan | RefusalReason | None:
    """Find the entry :func:`load` would read for ``key`` in ``scope``, by source mark.

    Composes the board with ``yaml.SafeLoader`` — the parser :func:`load` uses —
    and never serializes it: the nodes only say WHERE the row is, and
    :func:`_set_mapping_value` still owns every emitted byte. ``root`` is the
    document's top-level mapping; ``development_status`` is the mapping under
    that key. Text that merely looks like the row (inside a block scalar, under
    another mapping) is never a node of the right mapping, so it cannot be
    selected.

    The row is matched by the key :func:`load` READS, not by its node text
    (DW-515): :func:`_resolve_entry` constructs every key of the merge-flattened
    mapping and compares :func:`_key_identity` — so ``epic-01`` answers for
    ``epic-1`` — and the editor then rewrites the row under its own key text
    (``key_text``). Entries constructing to one key resolve last-wins, as the
    constructor's dict assignment does, so the row found here is the row
    :func:`story_status` read. Two or more DISTINCT constructed keys sharing the
    identity (``"2001-01-04"`` and the date ``2001-01-04``; ``epic-1`` and
    ``epic-01``) leave the reader's answer to iteration order, and come back as
    ``"key-ambiguous"``. The ``development_status`` section itself is still found
    by its last ``development_status`` node.

    Returns None when the mapping has no such entry: the key is absent or only
    reachable through a ``<<`` merge. An entry whose value is a collection, an
    alias to a node authored elsewhere, or a value carrying an anchor (other rows'
    aliases resolve to its text, so rewriting it would silently move them too), or
    an entry of a flow-style ``{...}`` mapping (its closing brace and separators
    share lines with values), IS the row, so it comes back as the
    :data:`RefusalReason` the writer must decline it with rather than as None —
    "not here" and "here, but not editable" are different answers. Raises
    :class:`SprintStatusError` when the lines do not parse, or a key does not
    construct — the caller parsed this board before editing it, so that can only
    be an edit gone wrong, and it must not be published."""
    text = "".join(lines)
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError as e:
        raise SprintStatusError(f"sprint status is not valid YAML: {e}") from e
    mapping = root
    if scope == "development_status":
        section = _last_entry(root, "development_status")
        mapping = section[1] if section else None
    entry = _resolve_entry(mapping, key, scope)
    if entry is None:
        return None
    if isinstance(entry, str):  # "key-ambiguous"
        return entry
    if any(isinstance(m, yaml.MappingNode) and m.flow_style for m in (root, mapping)):
        return "mapping-is-flow"
    key_node, value_node = entry
    if not isinstance(value_node, yaml.ScalarNode):
        return "value-not-scalar"
    if value_node.start_mark.index < key_node.end_mark.index:
        return "value-is-alias"  # its marks belong to the anchor, not to this row
    # A node's start mark is its first property (`&anchor`, `!tag`), so the
    # properties are the leading whitespace-separated `&`/`!` runs of its source.
    props = _NODE_PROPERTIES_RE.match(text, value_node.start_mark.index)
    if props and "&" in {p[0] for p in props.group().split()}:
        return "value-is-anchored"

    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    def where(index: int) -> tuple[int, int]:
        line = bisect.bisect_right(starts, index) - 1
        return line, index - starts[line]

    key_line, key_col = where(key_node.start_mark.index)
    first_line, start_col = where(value_node.start_mark.index)
    last_line, end_col = where(value_node.end_mark.index)
    return _RowSpan(
        key_line=key_line,
        key_col=key_col,
        value_first_line=first_line,
        value_start_col=start_col,
        value_last_line=last_line,
        value_end_col=end_col,
        style=value_node.style,
        value=value_node.value,
        key_text=key_node.value,
    )


# How :func:`load` tells one key from another: ``("epic", n)`` for an epic row,
# which it files by number, ``("key", text)`` for every other row the writer edits.
_KeyIdentity = tuple[str, str | int]


def _key_identity(key: object, scope: _Scope) -> _KeyIdentity:
    """The identity :func:`load` gives a ``development_status`` key, or the root.

    ``str()`` of the constructed key, then — under ``development_status`` only —
    an epic key collapses to its int number, as :func:`load` files it in
    ``epics``. The root (``last_updated``) is read by ``str()`` alone."""
    text = str(key)
    if scope == "development_status" and (m := EPIC_RE.match(text)):
        return ("epic", int(m.group(1)))
    return ("key", text)


def _resolve_entry(
    mapping: object, key: str, scope: _Scope
) -> tuple[yaml.Node, yaml.Node] | Literal["key-ambiguous"] | None:
    """The ``(key, value)`` node pair of ``mapping``'s entry the reader reads as ``key``.

    Mirrors the constructor (DW-515): merges are flattened first (merged entries
    before the mapping's own, as ``construct_mapping`` does), every scalar key is
    constructed, and the entries are collected into a dict keyed by the
    constructed value — so entries constructing to one key resolve last-wins,
    exactly as the reader's dict does, and the dict keeps the first key object it
    saw. Of those, the ones whose :func:`_key_identity` is ``key``'s are the
    candidates. More than one distinct candidate is ``"key-ambiguous"``: the
    reader's answer then rests on iteration order (:func:`load`'s ``epics`` keeps
    the last, :func:`story_status` the first), and no single row is certainly it.
    None when there is no candidate, or the one candidate came from a ``<<``
    merge — the mapping holds no entry of its own to edit.

    ``mapping`` is mutated by the flattening; the caller composed it for this one
    lookup. A merge or key the constructor refuses raises
    :class:`SprintStatusError`."""
    if not isinstance(mapping, yaml.MappingNode):
        return None
    own = {id(entry) for entry in mapping.value}
    constructor = yaml.constructor.SafeConstructor()
    entries: dict[object, tuple[yaml.Node, yaml.Node]] = {}
    try:
        constructor.flatten_mapping(mapping)
        for entry in mapping.value:
            key_node = entry[0]
            if not isinstance(key_node, yaml.ScalarNode):
                continue  # a collection key is unhashable: load() could not have read it
            entries[constructor.construct_object(key_node, deep=True)] = entry
    except yaml.YAMLError as e:
        raise SprintStatusError(f"sprint status key could not be read: {e}") from e
    target = _key_identity(key, scope)
    found = [e for k, e in entries.items() if _key_identity(k, scope) == target]
    if len(found) > 1:
        return "key-ambiguous"
    if not found or id(found[0]) not in own:
        return None
    return found[0]


def _last_entry(mapping: object, key: str) -> tuple[yaml.Node, yaml.Node] | None:
    """The ``(key, value)`` node pair of the LAST ``key`` entry in a mapping node."""
    if not isinstance(mapping, yaml.MappingNode):
        return None
    for key_node, value_node in reversed(mapping.value):
        if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
            return key_node, value_node
    return None


@dataclass(frozen=True)
class _Refused:
    """:func:`_set_mapping_value`'s answer for a row it found and would not rewrite."""

    reason: RefusalReason


# What one row edit did. The three plain answers and a refusal are distinct values,
# so no caller has to read "False" as whichever of them it happens to need.
_Edit = Literal["changed", "equal", "absent"] | _Refused


def _set_mapping_value(lines: list[str], key: str, new_value: str, *, scope: _Scope) -> _Edit:
    """In-place replace the value of the ``key`` entry in ``scope``, preserving
    indentation and any trailing ` # comment`. A minimal line edit (not a YAML
    round-trip) so the file's comments and structure — STATUS DEFINITIONS,
    WORKFLOW NOTES — survive verbatim.

    Which line is edited is decided by the parser, not by a text search:
    :func:`_locate_row` names the key line and the value's exact source span in
    ``scope`` (story and epic rows under ``development_status``, ``last_updated``
    at the root). A textual scan would take the first line that merely looks like
    the row — inside an earlier block scalar, under another mapping, or a
    duplicate the parser overrides — and rewrite it while the real row stays put.
    ``key`` names the row by the key the reader reads (DW-515), so ``epic-1``
    finds an ``epic-01:`` row, and the edit keeps that row's own key text. Two
    distinct keys reading as ``key`` refuse as ``key-ambiguous``.

    A one-line value is edited as a SPLICE of exactly its source span (DW-514,
    DW-516): the parser has already proved where the scalar starts and ends on
    the key line, so only ``[value_start_col, value_end_col)`` is replaced and
    every other byte — the gap after the colon, trailing whitespace, an inline
    comment, the line terminator — stays as authored. Nothing is guessed from
    the text: a `#` inside a quoted scalar (`"a # b"`) is inside the span, so it
    can never be promoted into a comment (#366), and a comment after a closing
    quote is carried rather than dropped. A plain value keeps the spaces this
    board needs (`last_updated: 01-06-2026 10:00`). A null value has an empty
    span right after the colon, so `key:  # note` becomes `key: done  # note`
    (a separating space is supplied only when there is no gap). A tag
    (`!!str backlog`) is part of the span and is replaced with the bare token. The
    gap and tail checks are defensive: text between the colon and the span that
    is not spaces or tabs, or anything after it but whitespace and an optional
    whitespace-led comment, refuses the row as ``unreadable-value``.

    Returns ``"changed"`` after a real edit, ``"equal"`` when the value is already
    ``new_value`` (idempotent, nothing edited), ``"absent"`` when ``scope`` holds no
    ``key`` entry, and :class:`_Refused` when the entry is there but the line edit
    cannot rewrite it exactly — a key authored in quotes, a row in a flow-style
    mapping, a value that is a collection, an alias, or anchored, a key line
    whose text around the value span is not gap and comment, a multi-line shape
    below. ``lines`` is untouched on every answer but
    ``"changed"``. Refused is deliberately not folded into absent: whether a
    declined row is an error is the caller's decision (:func:`_advance_locked`
    raises for the story row), and it can only make it on an answer that says
    which happened. Each line's terminator lies outside the span and is
    reattached exactly as authored (#576).

    A value may also run past the key line. The one such shape this reads is the
    FOLDED plain row a width-limited dump (ruamel wraps at 80) emits for a long
    key: nothing but whitespace after the colon, then a plain scalar starting on
    the very next line. :func:`_folded_span_is_plain` checks its lines; the key
    line and exactly the value's span collapse into one `key: value` line that
    takes the terminator the span's last line had. A value that STARTS on the key
    line is edited only when its span ends there too. Every other multi-line
    value — a block scalar (`|`, `>`), a quoted or plain scalar that wraps from
    the key line (whatever the indentation of its later lines), a nested mapping
    or sequence, a comment or blank line inside or before the value — is refused
    whole: rewriting only the key line would leave the rest behind as part of
    the value (`key: done` over `    backlog` parses as `done backlog`) or turn
    it into invalid YAML."""
    row = _locate_row(lines, key, scope)
    if row is None:
        return "absent"
    if not isinstance(row, _RowSpan):
        return _Refused(row)
    line = lines[row.key_line]
    stripped = line.rstrip("\r\n")
    km = re.match(rf"^(?P<indent>\s*){re.escape(row.key_text)}:", stripped)
    if km is None or len(km.group("indent")) != row.key_col:
        return _Refused("key-not-plain")
    indent, remainder = km.group("indent"), stripped[km.end() :]
    if row.value_first_line > row.key_line:
        if remainder.strip(" \t") or not _folded_span_is_plain(lines, row):
            return _Refused("multiline-value")  # not a shape this can read whole
        if row.value == new_value:
            return "equal"  # already at target — idempotent no-op
        last = lines[row.value_last_line]
        nl = last[len(last.rstrip("\r\n")) :]
        lines[row.key_line : row.value_last_line + 1] = [
            f"{indent}{row.key_text}: {new_value}" + nl
        ]
        return "changed"
    if row.value_last_line != row.key_line:
        # the value runs onto later lines — a one-line edit would orphan them
        return _Refused("multiline-value")
    if row.value_start_col < km.end():
        return _Refused("unreadable-value")
    gap = stripped[km.end() : row.value_start_col]
    tail = stripped[row.value_end_col :]
    if gap.strip(" \t") or not _VALUE_TAIL_RE.fullmatch(tail):
        return _Refused("unreadable-value")  # leave the line as authored
    if stripped[row.value_start_col : row.value_end_col] == new_value:
        return "equal"  # already at target — idempotent no-op
    nl = line[len(stripped) :]
    sep = "" if gap else " "
    lines[row.key_line] = stripped[: row.value_start_col] + sep + new_value + tail + nl
    return "changed"


# What may follow a one-line value's span on its key line: trailing spaces/tabs,
# or a comment, which YAML opens only at a whitespace-preceded `#`.
_VALUE_TAIL_RE = re.compile(r"[ \t]*|[ \t]+#.*")


# Characters that cannot open a line of the folded row this writer accepts:
# YAML indicators (block scalars, quotes, flow collections, anchors, tags,
# sequences, comments). A plain scalar's continuation line may legally open
# with most of them (`ready` over `- for dev` is the one value `ready - for
# dev`); refusing the whole class is stricter than YAML, and a refused row is
# left as authored.
_PLAIN_FRAGMENT_REFUSED_LEADS = frozenset("#|>'\"-?:,[]{}&*!%@`")


def _folded_span_is_plain(lines: list[str], row: _RowSpan) -> bool:
    """Is ``row``'s value the folded shape: a plain scalar starting on the line
    right after the key, every line of its span a content line opening with no
    indicator, and nothing but whitespace after it on its last line?

    The parser has already proved the span; this only narrows which spans the
    writer will collapse. A comment or blank line between key and value, a blank
    line inside it (YAML keeps it as a newline), or a comment after it would be
    lost by the collapse, so each refuses the row. So is a span ending in a
    Unicode line break (NEL, LS, PS): ``str.splitlines`` and YAML both end the
    line there, but it is not one of the ``\r``/``\n`` terminators the collapse
    carries over, so only spaces and tabs may follow the value."""
    if row.style is not None or row.value_first_line != row.key_line + 1:
        return False
    for text in lines[row.value_first_line : row.value_last_line + 1]:
        content = text.rstrip("\r\n").lstrip(" ")
        if not content.strip() or content[0] in _PLAIN_FRAGMENT_REFUSED_LEADS:
            return False
    tail = lines[row.value_last_line].rstrip("\r\n")[row.value_end_col :]
    return not tail.strip(" \t")


@contextmanager
def _board_lock(path: Path) -> Iterator[None]:
    """Cross-process mutual exclusion for one sprint-status board (#286/#469).

    The board's counterpart to :func:`~bmad_loop.deferredwork.ledger_lock`, and
    private for the same reason it is narrow: :func:`advance` is the only writer,
    so the only thing that ever needs to hold this is the read-modify-write below.
    Held around file I/O only — never across a subprocess, a coding-CLI session,
    or an operator pause (#286).

    The import of :mod:`~bmad_loop.runs` is lazy and has to stay lazy: ``runs``
    imports ``verify``, which imports this module, so a top-level import would
    close the cycle.
    """
    from . import runs

    with file_lock(runs.lock_path_for(path)):
        yield


def _row_at_or_past(current: str, target: str) -> bool:
    """Is a row at ``current`` already at or past ``target`` in :data:`STATUS_ORDER`?

    The never-regress comparison :func:`_advance_locked` makes, factored out so
    that :func:`advance`'s advisory pre-lock probe and the authoritative locked
    decision run one body and cannot drift apart (#736). A probe that answered
    this question even slightly differently from the writer would either skip a
    write the board needed or take a lock it did not.

    Deliberately NOT :func:`~bmad_loop.engine._at_or_past`, the reader-side twin:
    that one counts an exact match OUTSIDE ``STATUS_ORDER`` as reached, which is
    right for reading what :func:`advance` RETURNED and wrong as input to its
    WRITE decision. An off-order status equal to ``target`` is a no-op owned by
    :func:`_set_mapping_value` under the lock — it refuses a value it already
    holds — and routing it through this predicate instead would hand the answer
    to a pre-lock probe on a comparison the writer does not make.
    """
    return (
        current in STATUS_ORDER
        and target in STATUS_ORDER
        and STATUS_ORDER.index(current) >= STATUS_ORDER.index(target)
    )


def advance(
    path: Path,
    story_key: str,
    target: str,
    *,
    now: str | None = None,
    allow_regression: bool = False,
) -> str | None:
    """Advance a story's sprint-status to `target` for the generic-skill path.

    Mirrors sync-sprint-status.md: skip when the file is missing or the story is
    absent (returns None); never regress (returns the current status unchanged
    when it is already at or past `target` in STATUS_ORDER); lift a `backlog`
    parent epic to `in-progress` only when advancing a story to `in-progress`;
    refresh `last_updated` when `now` is given. Comments/structure are preserved
    via line edits. Returns the story's status after the call (== `target` on a
    write), or None when nothing was eligible.

    Raises :class:`SprintStatusWriteRefused` when the row exists, the call has
    decided to move it (below `target`, or an allowlisted regression), and the
    line edit cannot rewrite its shape — a quoted key, an alias, a block scalar.
    Nothing is written then, not even the epic lift or `last_updated`. This used
    to return the unchanged status, which a caller cannot tell from a story that
    never got there; the raise names the board, the row, and why (#842).

    The rewrite is atomic and symlink-following (#379), and every existing CRLF,
    LF, bare CR, or mixed per-line terminator is preserved (#576). The board is
    read as raw UTF-8 bytes, each edited line carries its own terminator, and
    `atomic_write_bytes` publishes the byte-exact result. This is a
    read-modify-rewrite of the board, and a truncating write that faults partway
    through corrupts it SILENTLY: YAML cut at a line boundary is still a valid
    mapping, just a smaller one, so the epics past the tear cease to exist rather
    than raising. AGENTS.md makes this the orchestrator's sole write path to
    sprint-status.yaml, so nothing downstream would contradict the shortened
    board — the run would simply walk off the end of the sprint. The atomic
    helper keeps the file entire: either the old contents or the whole new ones,
    never a prefix.

    Symlinks are FOLLOWED (the helper's default), which is what the old
    truncating write did too — the board is an operator-curated file at a
    project-relative path, and a repo that symlinks it somewhere must keep being
    a symlink. That rules out the confined writers, which are no-follow by
    construction: this site takes the #597 flag and nothing else.

    ``require_writable_target=True`` is that flag, and it restores what going
    atomic silently dropped: `os.replace` needs write permission on the parent
    DIRECTORY, never on the entry it replaces, so a board an operator had marked
    read-only was rewritten anyway — and because the mode is inherited it came
    back reading ``0444``, leaving nothing in the permission bits to record that
    it changed (#597). The truncating `write_bytes` this replaced raised
    `PermissionError` there as a side effect of opening the file; that refusal is
    a property worth keeping deliberately, because AGENTS.md makes this the
    orchestrator's SOLE write path to the board — a read-only board is the only
    way an operator can say "stop rewriting this", and it has to mean something.

    Serialized cross-process (#286/#469) on the board's advisory lock — the
    state-root sidecar :func:`~bmad_loop.runs.lock_path_for` names for it, not a
    sibling of the board itself, because the board is a tracked file and the
    engine's own ``git add -A`` would commit a sidecar beside it. The hold spans
    the whole read-modify-write and nothing else: three reads (the status probe,
    the raw bytes, the epic-lift ``load``) and the one atomic write, with no
    subprocess, session, or operator pause inside it (#286). That also closes the
    intra-call TOCTOU, since the never-regress decision and the bytes it is
    applied to now come from one hold rather than from two independent reads.

    Two answers are reached BEFORE the lock. The missing-board check runs first,
    so asking about a board that does not exist leaves no sidecar behind. Then an
    ADVISORY probe (#736) reads the row once and answers the two cases in which
    this call would write nothing at all: an absent row (``None``) and a row
    already at or past ``target`` (the current status, via :func:`_row_at_or_past`
    — the same predicate the locked body applies). Acquiring for those was the
    defect: an idempotent replay — ``bmad-loop confirm`` against a story the board
    already records as done is a designed path, not an error
    (:meth:`~bmad_loop.model.ParkedStory.resumable` accepts it), as is
    ``_carry_board_advance``'s routine no-op on a tracked board — could fail on
    lock contention, or on a :class:`~bmad_loop.runs.StateRootError` from
    :func:`~bmad_loop.runs.lock_path_for`, for work it was never going to do.

    The probe is advisory in the strict sense: only a "would write nothing"
    answer is acted on, and such a call simply linearizes at the probe's read
    rather than at an acquisition. Every other outcome — including ANY exception
    raised while probing — falls through to the locked path, which re-reads,
    re-decides authoritatively and raises on the channel it always did. So the
    probe can neither authorize a write nor add a failure mode the hold lacks: a
    malformed board still raises :class:`SprintStatusError` from under the lock,
    and a write refusal is decided there too, never by the probe.
    ``now`` needs no handling here, because both no-op arms of
    :func:`_advance_locked` return before the ``last_updated`` write; a
    probe-satisfied early-out is write-equivalent to the locked answer.

    Acquisition failure — for the calls that do reach the lock — surfaces as
    ``OSError`` (or :class:`~bmad_loop.runs.StateRootError` when no state root can
    be derived) on the channel callers already route this function's raises
    through — the engine's crash/escalation handling, the CLI's failure exit — so
    a board that could not be serialized fails loudly rather than being rewritten
    unlocked. :func:`advanced_bytes` deliberately does NOT come through
    here: it calls :func:`_advance_locked` against a private throwaway copy, so it
    neither contends on the real board's sidecar nor mints one of its own.

    ``allow_regression=True`` is the one allowlisted exception to never-regress
    (DW-383). A row strictly PAST ``target`` is then moved back to it when the
    ``(current, target)`` pair is in
    :data:`~bmad_loop.statemachine.BOARD_REGRESSIONS`; any other regressing pair
    raises :class:`~bmad_loop.statemachine.IllegalTransition` and writes nothing.
    Forward moves, a row already AT ``target``, and absent rows behave exactly as
    without the flag. The advisory probe never answers a regression the flag asks
    for — it falls through to the lock, and the allowlist check runs there, under
    the hold, because the probe swallows every exception it meets.
    :func:`advanced_bytes` never passes the flag.
    """
    if not path.is_file():
        return None  # no board, nothing to serialize against — take no lock
    try:
        current = story_status(path, story_key)
        if current is None:
            return None  # absent row — nothing this call would write
        if _row_at_or_past(current, target) and not (allow_regression and current != target):
            return current  # already at or past target — never regress, no write
    except Exception:  # nosec B110 - ADVISORY probe: a fault here must decide nothing
        # Broad by design, and the swallow is the point: narrowing the catch would
        # let the probe invent a failure mode the locked path does not have. An
        # unreadable board decides nothing here — the path below re-reads,
        # re-decides, and raises on the channel callers already route this
        # function's raises through.
        pass
    with _board_lock(path):
        return _advance_locked(path, story_key, target, now=now, allow_regression=allow_regression)


def _advance_locked(
    path: Path,
    story_key: str,
    target: str,
    *,
    now: str | None = None,
    allow_regression: bool = False,
) -> str | None:
    """:func:`advance`'s read-modify-write, run with the board's lock already held.

    Split out so the hold is exactly the file I/O and so every read inside it sees
    one board. The reads below repeat work the caller's pre-lock answers may
    already have done, and deliberately: those answers are taken without
    exclusion, so a delete can land between the ``is_file`` check and the
    acquisition, and the advisory probe's row (#736) can be stale by the time the
    lock is held. Only what this function reads decides the published bytes.

    ``allow_regression`` is :func:`advance`'s opt-in: the authoritative
    allowlist check for a regressing pair happens here, under the hold, and a
    refused pair raises before any byte is written.

    So does the write refusal: :class:`SprintStatusWriteRefused` is decided here
    and only here, from this hold's own reread, once that read has put the row
    below ``target`` (or on an allowlisted regression) and the story edit then
    comes back refused — or absent, when the reader resolved the key through a
    ``<<`` merge no entry of ``development_status`` holds. It raises before the
    epic lift and ``last_updated`` are touched and before the write, so a refused
    call publishes nothing. :func:`advance`'s pre-lock probe cannot reach it: the
    probe only ever answers "nothing to write", and swallows whatever it meets.

    The epic lift is best effort, save one answer: an epic edit refused as
    ``key-ambiguous`` (DW-515) raises :class:`SprintStatusWriteRefused` for the
    epic's canonical key, before the write, so the story row is not published
    either — lifting whichever row the writer picked could regress the one the
    reader does not report."""
    if not path.is_file():
        return None
    current = story_status(path, story_key)
    if current is None:
        return None
    if _row_at_or_past(current, target):
        if not allow_regression or current == target:
            return current  # already at or past target — never regress
        check_board_regression(story_key, current, target)  # raises unless allowlisted

    text = path.read_bytes().decode("utf-8")
    lines = text.splitlines(keepends=True)
    # story_status() resolves keys via a full YAML parse, but _set_mapping_value
    # rewrites via a line edit that can't touch every shape it finds (quoted or
    # block-scalar keys). A story row this call has decided to move and cannot
    # rewrite raises rather than echoing its unchanged status: a caller reading
    # that echo sees only "not at target" and blames whoever was supposed to get
    # it there. Each call below re-locates its row in the lines as edited so far,
    # so an earlier collapse never leaves a later edit aiming at a stale line.
    story_edit = _set_mapping_value(lines, story_key, target, scope="development_status")
    if isinstance(story_edit, _Refused):
        raise SprintStatusWriteRefused(path, story_key, current, target, story_edit.reason)
    if story_edit == "absent":
        raise SprintStatusWriteRefused(path, story_key, current, target, "row-not-in-mapping")
    if story_edit == "equal":
        return current  # an off-order value already spelled as `target` — nothing to write

    if target == "in-progress":
        m = STORY_RE.match(story_key)
        if m:
            epic_key = f"epic-{int(m.group(1))}"
            ss = load(path)
            epic_status = ss.epics.get(int(m.group(1)))
            if epic_status == "backlog":
                # best effort, as before: an epic row the writer declines stays put —
                # unless two rows read as this epic, which nothing could lift safely
                epic_edit = _set_mapping_value(
                    lines, epic_key, "in-progress", scope="development_status"
                )
                if epic_edit == _Refused("key-ambiguous"):
                    raise SprintStatusWriteRefused(
                        path, epic_key, epic_status, "in-progress", "key-ambiguous"
                    )

    if now is not None:
        _set_mapping_value(lines, "last_updated", now, scope="root")  # best effort, likewise

    atomic_write_bytes(path, "".join(lines).encode("utf-8"), require_writable_target=True)
    return target


def advanced_bytes(source: bytes, story_key: str, target: str) -> bytes | None:
    """What :func:`advance` would leave behind, given ``source`` as the board's bytes.

    For the caller that has to know whether a board on disk holds THIS run's advance
    and nothing else, and so needs the intended content recomputed from a baseline it
    trusts rather than read back out of the file it is about to commit.

    Goes through the real writer's own body (``_advance_locked``), against a
    throwaway copy, rather than reimplementing the edit. Never-regress, the epic lift, and
    ``_set_mapping_value``'s quoted-scalar, inline-comment and per-line-terminator
    handling ARE what makes two boards "the same advance" — a second implementation of
    them would drift from the writer silently, and for this caller a silent drift means
    committing somebody else's bytes.

    No ``now=``: the caller's own carry passes none either, and a ``last_updated`` line
    rewritten here and not there would make every comparison fail.

    Returns None only when the board's row is absent. The other None — a missing
    file — cannot be reached from here, the shadow being this function's own
    copy. There is then no intended content to compare against, and a caller must not
    read "I could not compute it" as "the tree is mine".

    A row already at or past ``target`` is the one decline that comes back as bytes:
    ``advance`` writes nothing for it, so ``source`` byte-identical IS this run's
    advance, and a caller is right to accept an untouched board.

    A row below ``target`` whose line ``_set_mapping_value`` will not rewrite (or
    whose epic lift is ``key-ambiguous``, ``story_key`` then ``epic-N``) is not
    that case, and raises :class:`SprintStatusWriteRefused` (with ``path`` None — the
    shadow is no board anyone can repair). ``advance`` raises there too, so no board
    on disk holds that advance; returning ``source`` would let a caller match an
    untouched — or someone else's — board against an advance that never happened and
    claim it. A caller computing ownership must treat the raise as "not mine", as it
    does None. A board that does not parse raises :class:`SprintStatusError`, as
    ``advance`` does."""
    with tempfile.TemporaryDirectory() as tmp:
        shadow = Path(tmp) / "sprint-status.yaml"
        shadow.write_bytes(source)
        # Straight to the locked body, deliberately skipping `advance`'s own
        # acquisition. The shadow is this function's private copy inside a
        # TemporaryDirectory that no other process can name, so there is no
        # second writer to exclude — and taking the lock anyway would mint a
        # state-root sidecar keyed on a path that exists only for this call.
        # `file_lock` never removes a sidecar and the TemporaryDirectory removes
        # only the shadow, so every ownership computation would strand another
        # dead lock file under `<state root>/locks` (#286).
        try:
            if _advance_locked(shadow, story_key, target) is None:
                return None
        except SprintStatusWriteRefused as e:
            raise SprintStatusWriteRefused(None, e.story_key, e.current, e.target, e.reason) from e
        return shadow.read_bytes()


def status_in_bytes(source: bytes, story_key: str) -> str | None:
    """:func:`story_status` asked of a board held as BYTES rather than as a file.

    For the caller comparing a live row against the same row at a git revision,
    where one side is a blob and never a path on disk.

    Goes through ``story_status`` against a throwaway copy, like
    :func:`advanced_bytes` above and for its reason: the full YAML resolution and the
    ``LEGACY_STORY_STATUSES`` folding ARE what makes two rows "the same status", and a
    second reading of them would drift from the one every other caller uses.

    Returns None when the row is absent. A board that does not parse raises
    ``SprintStatusError``, exactly as ``story_status`` does — "I could not read it"
    must not reach a caller spelled as "the row is gone".
    """
    with tempfile.TemporaryDirectory() as tmp:
        shadow = Path(tmp) / "sprint-status.yaml"
        shadow.write_bytes(source)
        return story_status(shadow, story_key)


@dataclass(frozen=True)
class StorySelector:
    """Resolves a human story reference (``--epic``/``--story``) to the
    stories it selects. Forms accepted by :func:`parse_selector`:

    * full key ``3-1-user-auth`` — exact match
    * short ref ``3-1`` / ``3.1`` — epic 3, story 1 (any slug)
    * suffixed short ref ``2-6a`` / ``2.6a`` — exactly the ``a`` half of a
      split story; the plain ``2-6`` matches the whole ``2-6a``/``2-6b`` family
    * bare number ``1`` (or ``6a``) with ``--epic 3`` — epic 3, story 1 (or 6a)
    * slug fragment ``user-auth`` / ``auth`` — substring of the slug (must be unique)
    * epic only (``--epic 3``, blank story) — every story in the epic
    """

    epic: int | None = None
    num: int | None = None
    key: str | None = None  # exact full key
    slug: str | None = None  # slug substring
    suffix: str | None = None  # split-story letter; None matches any suffix

    @property
    def is_targeted(self) -> bool:
        """True when the selector names one intended story rather than just
        an epic-wide (or empty) filter."""
        return any(v is not None for v in (self.key, self.num, self.slug))

    def matches(self, story: Story) -> bool:
        if self.key is not None:
            return story.key == self.key
        if self.epic is not None and story.epic != self.epic:
            return False
        if self.num is not None and story.num != self.num:
            return False
        if self.suffix is not None and story.suffix != self.suffix:
            return False
        if self.slug is not None and self.slug not in story.slug:
            return False
        return True


def parse_selector(epic: int | None, story: str | None) -> StorySelector:
    """Translate the ``--epic``/``--story`` pair into a :class:`StorySelector`.

    Raises :class:`SprintStatusError` on bad or ambiguous input.
    """
    text = (story or "").strip()
    if not text:
        return StorySelector(epic=epic)

    def _check_epic(parsed_epic: int) -> None:
        if epic is not None and epic != parsed_epic:
            raise SprintStatusError(
                f"--epic {epic} conflicts with story '{text}' (epic {parsed_epic})"
            )

    # empty suffix group -> None: a plain `2-6` matches the whole split family
    if m := STORY_RE.match(text):  # full key 3-1-slug
        e, n = int(m.group(1)), int(m.group(2))
        _check_epic(e)
        return StorySelector(epic=e, num=n, key=text, suffix=m.group(3) or None)
    if m := SHORT_REF_RE.match(text):  # 3-1 / 3.1 / 3-1a
        e, n = int(m.group(1)), int(m.group(2))
        _check_epic(e)
        return StorySelector(epic=e, num=n, suffix=m.group(3) or None)
    if m := BARE_NUM_RE.match(text):  # bare story number, needs --epic
        if epic is None:
            raise SprintStatusError(
                f"ambiguous story '{text}': use --epic E --story {text}, or E-{text}"
            )
        return StorySelector(epic=epic, num=int(m.group(1)), suffix=m.group(2) or None)
    return StorySelector(epic=epic, slug=text)  # slug fragment


def select_actionable(ss: SprintStatus, epic: int | None, story: str | None) -> list[Story]:
    """Stories selected by ``--epic``/``--story`` that are ready to start, in
    file order. Raises :class:`SprintStatusError` with a targeted message when a
    named story is missing, ambiguous, or exists but is not actionable.
    """
    sel = parse_selector(epic, story)
    matches = [s for s in ss.stories if sel.matches(s)]
    if sel.is_targeted:
        if not matches:
            raise SprintStatusError(f"no story matches '{story}'")
        if sel.slug is not None:
            keys = sorted({s.key for s in matches})
            if len(keys) > 1:
                raise SprintStatusError(
                    f"story '{sel.slug}' is ambiguous — matches: {', '.join(keys)}"
                )
    actionable = [s for s in matches if s.status in ACTIONABLE_STATUSES]
    if sel.is_targeted and matches and not actionable:
        s = matches[0]
        raise SprintStatusError(
            f"story {story} matched {s.key} but its status is " f"'{s.status}' (not actionable)"
        )
    return actionable
