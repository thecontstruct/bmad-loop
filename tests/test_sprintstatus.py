import pytest
from conftest import fault_metadata_probe, write_sprint

from bmad_loop import sprintstatus


def test_load_classifies_keys(project_tree):
    write_sprint(
        project_tree,
        {
            "epic-1": "in-progress",
            "1-1-user-auth": "done",
            "1-2-account-mgmt": "ready-for-dev",
            "epic-1-retrospective": "optional",
            "epic-2": "backlog",
            "2-1-personality": "backlog",
            "epic-2-retrospective": "optional",
            "epic-1-retro-item-1-test-design": "done",
            "epic-2-retro-item-3-fts5-research": "backlog",
            "weird-key": "huh",
        },
    )
    ss = sprintstatus.load(project_tree.sprint_status)
    assert ss.epics == {1: "in-progress", 2: "backlog"}
    assert [s.key for s in ss.stories] == [
        "1-1-user-auth",
        "1-2-account-mgmt",
        "2-1-personality",
    ]
    assert ss.stories[1].epic == 1 and ss.stories[1].num == 2
    assert ss.retros == {1: "optional", 2: "optional"}
    assert ss.unknown_keys == ("weird-key",)


def test_load_split_story_keys(project_tree):
    # BMAD splits an oversized story into 2-6a / 2-6b (issue #144); both halves
    # must parse as stories — not fall into unknown_keys — and keep file order.
    write_sprint(
        project_tree,
        {
            "2-5-intact": "done",
            "2-6a-build-structure": "backlog",
            "2-6b-extend-structure": "backlog",
            "2-7-later": "backlog",
            "2-6ab-not-a-split": "backlog",  # multi-letter: not the convention
            "2-6A-not-lower": "backlog",  # uppercase: not the convention
        },
    )
    ss = sprintstatus.load(project_tree.sprint_status)
    assert [s.key for s in ss.stories] == [
        "2-5-intact",
        "2-6a-build-structure",
        "2-6b-extend-structure",
        "2-7-later",
    ]
    a, b = ss.stories[1], ss.stories[2]
    assert (a.epic, a.num, a.suffix, a.slug) == (2, 6, "a", "build-structure")
    assert (b.epic, b.num, b.suffix, b.slug) == (2, 6, "b", "extend-structure")
    assert ss.stories[0].suffix == ""  # whole stories carry no suffix
    assert ss.unknown_keys == ("2-6ab-not-a-split", "2-6A-not-lower")
    # the split halves are picked in file order, before later stories
    assert sprintstatus.next_actionable(ss).key == "2-6a-build-structure"


def test_load_classifies_retro_items(project_tree):
    write_sprint(
        project_tree,
        {
            "epic-1-retrospective": "done",
            "epic-1-retro-item-1-test-design-in-stories": "done",
            "epic-5-retro-item-2-singleflight-inflight-guard-helper": "backlog",
        },
    )
    ss = sprintstatus.load(project_tree.sprint_status)
    # retro action items are recognized, not dumped into unknown_keys
    assert ss.unknown_keys == ()
    assert ss.retros == {1: "done"}  # plain retrospective key is unaffected
    assert [(r.key, r.epic, r.num, r.slug, r.status) for r in ss.retro_items] == [
        ("epic-1-retro-item-1-test-design-in-stories", 1, 1, "test-design-in-stories", "done"),
        (
            "epic-5-retro-item-2-singleflight-inflight-guard-helper",
            5,
            2,
            "singleflight-inflight-guard-helper",
            "backlog",
        ),
    ]


def test_retro_items_do_not_become_actionable_stories(project_tree):
    # recognition only: retro items must not leak into story selection
    write_sprint(project_tree, {"epic-3-retro-item-1-do-a-thing": "backlog"})
    ss = sprintstatus.load(project_tree.sprint_status)
    assert ss.stories == ()
    assert sprintstatus.next_actionable(ss) is None


def test_legacy_drafted_maps_to_ready(project_tree):
    write_sprint(project_tree, {"1-1-x": "drafted"})
    ss = sprintstatus.load(project_tree.sprint_status)
    assert ss.stories[0].status == "ready-for-dev"


def test_next_actionable_order_and_skip(project_tree):
    write_sprint(
        project_tree,
        {"1-1-a": "done", "1-2-b": "ready-for-dev", "1-3-c": "backlog"},
    )
    ss = sprintstatus.load(project_tree.sprint_status)
    assert sprintstatus.next_actionable(ss).key == "1-2-b"
    assert sprintstatus.next_actionable(ss, skip={"1-2-b"}).key == "1-3-c"
    assert sprintstatus.next_actionable(ss, skip={"1-2-b", "1-3-c"}) is None


def test_awaiting_operator_is_ordered_just_below_done():
    """The token's whole contract is its POSITION: last stop before `done`, so
    confirming a parked story is a forward advance and nothing can regress `done`
    back into it. Asserting the index relationships rather than the literal tuple
    keeps this from breaking every time a status is added elsewhere."""
    order = sprintstatus.STATUS_ORDER
    assert order.index("awaiting-operator") == order.index("done") - 1
    assert order.index("review") < order.index("awaiting-operator")


def test_parked_story_is_never_picked_as_next_actionable(project):
    """A parked story's agent-doable work is already committed — picking it up
    again would redo finished work while the human's external actions stay
    outstanding. The board must walk straight past it to the next real story.

    Ablation (repo rule, inverse form — the gate here is an ABSENCE, so the
    ablation ADDS rather than deletes): put "awaiting-operator" into
    ACTIONABLE_STATUSES and this test must fail on the first assert. It does —
    which is what proves the assertions gate on actionability rather than on
    file order (1-1-a is first in the file, so a test that merely returned the
    first pickable story would pass either way).
    """
    write_sprint(
        project,
        {"1-1-a": "awaiting-operator", "1-2-b": "ready-for-dev"},
    )
    ss = sprintstatus.load(project.sprint_status)
    assert sprintstatus.next_actionable(ss).key == "1-2-b"
    # and with the only other story taken, there is nothing left to run — the
    # parked story is not a fallback either
    assert sprintstatus.next_actionable(ss, skip={"1-2-b"}) is None


def test_parked_story_cannot_be_targeted_by_a_selector(project_tree):
    """`--story` naming a parked story is a user error with a specific message,
    not a silent re-drive: select_actionable filters on the same set."""
    write_sprint(project_tree, {"1-1-a": "awaiting-operator"})
    ss = sprintstatus.load(project_tree.sprint_status)
    with pytest.raises(
        sprintstatus.SprintStatusError,
        match=r"story 1-1 matched 1-1-a but its status is 'awaiting-operator'",
    ):
        sprintstatus.select_actionable(ss, None, "1-1")


def test_next_actionable_epic_filter(project_tree):
    # document order (epic 5 before epic 9), not numeric; the epic filter returns
    # epic 9's first actionable story even though 5-1 is earlier in the file.
    write_sprint(
        project_tree,
        {"5-1-e5": "backlog", "9-0-x": "ready-for-dev", "9-1-y": "backlog"},
    )
    ss = sprintstatus.load(project_tree.sprint_status)
    assert sprintstatus.next_actionable(ss).key == "5-1-e5"  # unfiltered = file order
    assert sprintstatus.next_actionable(ss, epic=9).key == "9-0-x"
    assert sprintstatus.next_actionable(ss, skip={"9-0-x"}, epic=9).key == "9-1-y"
    assert sprintstatus.next_actionable(ss, skip={"9-0-x", "9-1-y"}, epic=9) is None


def test_story_status_reread(project_tree):
    write_sprint(project_tree, {"1-1-a": "in-progress"})
    assert sprintstatus.story_status(project_tree.sprint_status, "1-1-a") == "in-progress"
    assert sprintstatus.story_status(project_tree.sprint_status, "9-9-z") is None


def test_parse_selector_forms():
    # full key — exact match intent
    sel = sprintstatus.parse_selector(None, "3-1-user-auth")
    assert (sel.epic, sel.num, sel.key, sel.slug) == (3, 1, "3-1-user-auth", None)
    # short refs: hyphen and dot are equivalent
    for ref in ("3-1", "3.1"):
        sel = sprintstatus.parse_selector(None, ref)
        assert (sel.epic, sel.num, sel.key, sel.slug) == (3, 1, None, None)
    # bare number resolves against --epic
    sel = sprintstatus.parse_selector(3, "1")
    assert (sel.epic, sel.num, sel.slug) == (3, 1, None)
    # slug fragment
    sel = sprintstatus.parse_selector(None, "user-auth")
    assert (sel.epic, sel.num, sel.slug) == (None, None, "user-auth")
    # epic only — not targeted
    sel = sprintstatus.parse_selector(3, None)
    assert sel.epic == 3 and not sel.is_targeted


def test_parse_selector_split_suffix():
    # every numeric form carries the split suffix through to the selector
    sel = sprintstatus.parse_selector(None, "2-6a-build-structure")
    assert (sel.epic, sel.num, sel.suffix, sel.key) == (2, 6, "a", "2-6a-build-structure")
    for ref in ("2-6a", "2.6a"):
        sel = sprintstatus.parse_selector(None, ref)
        assert (sel.epic, sel.num, sel.suffix, sel.key, sel.slug) == (2, 6, "a", None, None)
    sel = sprintstatus.parse_selector(2, "6a")
    assert (sel.epic, sel.num, sel.suffix) == (2, 6, "a")
    # suffix-less forms leave suffix None — the whole-family wildcard
    for epic, ref in [(None, "2-6"), (None, "2.6"), (2, "6"), (None, "2-6-whole-slug")]:
        sel = sprintstatus.parse_selector(epic, ref)
        assert sel.suffix is None, ref


def test_parse_selector_bare_number_needs_epic():
    with pytest.raises(sprintstatus.SprintStatusError, match="ambiguous story '1'"):
        sprintstatus.parse_selector(None, "1")


def test_parse_selector_epic_conflict():
    with pytest.raises(sprintstatus.SprintStatusError, match="conflicts"):
        sprintstatus.parse_selector(2, "3-1")
    with pytest.raises(sprintstatus.SprintStatusError, match="conflicts"):
        sprintstatus.parse_selector(2, "3-1-user-auth")


def test_select_actionable_short_ref_and_epic_story(project_tree):
    write_sprint(
        project_tree,
        {"3-1-user-auth": "ready-for-dev", "3-2-foo": "backlog", "4-1-bar": "backlog"},
    )
    ss = sprintstatus.load(project_tree.sprint_status)
    for epic, story in [(None, "3-1"), (None, "3.1"), (3, "1"), (None, "user-auth")]:
        got = sprintstatus.select_actionable(ss, epic, story)
        assert [s.key for s in got] == ["3-1-user-auth"]
    # epic only selects every actionable story in the epic
    assert [s.key for s in sprintstatus.select_actionable(ss, 3, None)] == [
        "3-1-user-auth",
        "3-2-foo",
    ]


def test_select_actionable_split_suffix(project_tree):
    write_sprint(
        project_tree,
        {
            "2-6a-build-structure": "backlog",
            "2-6b-extend-structure": "backlog",
            "2-7-later": "backlog",
        },
    )
    ss = sprintstatus.load(project_tree.sprint_status)
    # a suffixed ref selects exactly its half — never the sibling
    for epic, story in [(None, "2-6a"), (None, "2.6a"), (2, "6a")]:
        got = sprintstatus.select_actionable(ss, epic, story)
        assert [s.key for s in got] == ["2-6a-build-structure"]
    assert [s.key for s in sprintstatus.select_actionable(ss, None, "2.6b")] == [
        "2-6b-extend-structure"
    ]
    # the plain short ref selects the whole split family, in file order
    assert [s.key for s in sprintstatus.select_actionable(ss, None, "2-6")] == [
        "2-6a-build-structure",
        "2-6b-extend-structure",
    ]
    # a suffix that doesn't exist matches nothing
    with pytest.raises(sprintstatus.SprintStatusError, match="no story matches '2-6c'"):
        sprintstatus.select_actionable(ss, None, "2-6c")


def test_select_actionable_split_suffix_not_actionable(project_tree):
    write_sprint(
        project_tree,
        {"2-6a-build-structure": "done", "2-6b-extend-structure": "backlog"},
    )
    ss = sprintstatus.load(project_tree.sprint_status)
    with pytest.raises(
        sprintstatus.SprintStatusError,
        match=r"story 2-6a matched 2-6a-build-structure but its status is 'done'",
    ):
        sprintstatus.select_actionable(ss, None, "2-6a")
    # the family ref still finds the remaining actionable half
    got = sprintstatus.select_actionable(ss, None, "2-6")
    assert [s.key for s in got] == ["2-6b-extend-structure"]


def test_select_actionable_targeted_not_actionable(project_tree):
    write_sprint(project_tree, {"3-1-user-auth": "ready-for-dev", "3-2-foo": "done"})
    ss = sprintstatus.load(project_tree.sprint_status)
    with pytest.raises(
        sprintstatus.SprintStatusError,
        match=r"story 3-2 matched 3-2-foo but its status is 'done'",
    ):
        sprintstatus.select_actionable(ss, None, "3-2")


def test_select_actionable_ambiguous_slug(project_tree):
    write_sprint(project_tree, {"3-1-user-auth": "backlog", "4-2-admin-auth": "backlog"})
    ss = sprintstatus.load(project_tree.sprint_status)
    with pytest.raises(sprintstatus.SprintStatusError, match="ambiguous"):
        sprintstatus.select_actionable(ss, None, "auth")


def test_select_actionable_no_match(project_tree):
    write_sprint(project_tree, {"3-1-user-auth": "backlog"})
    ss = sprintstatus.load(project_tree.sprint_status)
    with pytest.raises(sprintstatus.SprintStatusError, match="no story matches"):
        sprintstatus.select_actionable(ss, None, "9-9")


def test_missing_file_raises(project_tree):
    with pytest.raises(sprintstatus.SprintStatusError, match="not found"):
        sprintstatus.load(project_tree.sprint_status)


def test_malformed_yaml_raises(project_tree):
    project_tree.sprint_status.write_text("development_status: [unclosed")
    with pytest.raises(sprintstatus.SprintStatusError, match="not valid YAML"):
        sprintstatus.load(project_tree.sprint_status)


def test_missing_map_raises(project_tree):
    project_tree.sprint_status.write_text("project: x\n")
    with pytest.raises(sprintstatus.SprintStatusError, match="development_status"):
        sprintstatus.load(project_tree.sprint_status)


# ---------------------------------------------------------------- action_items (DW-388)

_BOARD_WITH_ITEMS = """\
development_status:
  epic-1: done
  1-1-user-auth: done
  epic-1-retrospective: done
  epic-1-retro-item-1-test-design: backlog

action_items:
  - id: "epic-1-retro-item-1-add-x"
    epic: 1
    action: "Add X to the checklist"
    owner: "Amelia"
    status: open
    ref: "docs/retro-epic-1.md"
  - epic: 1
    action: "legacy item without an id"
    owner: "Charlie"
    status: open
  - "not a mapping"
  - id: "   "
    action: "blank id"
  - id: 7
    action: "non-string id"
  - id: "  epic-1-retro-item-2-done-thing  "
    epic: "1"
    action: 42
    owner: ""
    status: done
    ref: ""
  - id: epic-2-retro-item-1-bare
  - id: epic-3-retro-item-1-superscript
    epic: "²"
  - id: epic-3-retro-item-2-bool
    epic: true
"""


def test_load_action_items_reads_id_keyed_items_in_file_order(project_tree):
    """The happy path, plus every skip rule and normalization in one board.

    Ablation: drop the `if item_id is None: continue` skip and the legacy row (and
    the blank/non-string ids) appear, reddening the id list."""
    project_tree.sprint_status.write_text(_BOARD_WITH_ITEMS, encoding="utf-8")
    items = sprintstatus.load_action_items(project_tree.sprint_status)
    assert items is not None
    assert [i.id for i in items] == [
        "epic-1-retro-item-1-add-x",
        "epic-1-retro-item-2-done-thing",  # stripped
        "epic-2-retro-item-1-bare",
        "epic-3-retro-item-1-superscript",
        "epic-3-retro-item-2-bool",
    ]
    first, done, bare, superscript, boolean = items
    assert first == sprintstatus.ActionItem(
        id="epic-1-retro-item-1-add-x",
        epic=1,
        action="Add X to the checklist",
        owner="Amelia",
        status="open",
        ref="docs/retro-epic-1.md",
    )
    # a digit-string epic is read as a number; a non-string action is None, and a
    # blank owner / ref is absent rather than an empty string
    assert (done.epic, done.action, done.owner, done.status, done.ref) == (
        1,
        None,
        None,
        "done",
        None,
    )
    # nothing but an id: everything optional absent, the status empty
    assert (bare.epic, bare.action, bare.owner, bare.status, bare.ref) == (
        None,
        None,
        None,
        "",
        None,
    )
    # "²" is a digit to `isdigit` but not a decimal `int()` accepts, and `true`
    # is an `int` subclass that names no epic: neither may raise or become a number
    assert superscript.epic is None
    assert boolean.epic is None


def test_load_action_items_absent_board_is_none(project_tree):
    """Absence is not a fault: no board means nothing to ingest.

    Ablation: drop the `FileNotFoundError` arm and this raises
    `SprintStatusError` instead of answering None."""
    project_tree.sprint_status.unlink(missing_ok=True)
    assert sprintstatus.load_action_items(project_tree.sprint_status) is None


@pytest.mark.parametrize(
    "board",
    [
        "development_status:\n  epic-1: backlog\n",
        # no development_status either: the retro list is read independently of it
        "project: x\n",
        # an empty `action_items:` is what the retrospective's own writer treats as []
        "development_status:\n  epic-1: backlog\naction_items:\n",
        "action_items: []\n",
    ],
    ids=["no-key", "no-dev-status", "null-value", "empty-list"],
)
def test_load_action_items_without_items_is_empty(project_tree, board):
    project_tree.sprint_status.write_text(board, encoding="utf-8")
    assert sprintstatus.load_action_items(project_tree.sprint_status) == ()


@pytest.mark.parametrize(
    "board",
    ["action_items: {id: x}\n", "action_items: just text\n", "action_items: 3\n"],
    ids=["mapping", "scalar", "number"],
)
def test_load_action_items_non_list_raises_malformed(project_tree, board):
    """Ablation: drop the `isinstance(listed, list)` check and the mapping row
    iterates its keys (skipping them all) and returns `()` silently, the scalar
    row iterates characters — none raise."""
    project_tree.sprint_status.write_text(board, encoding="utf-8")
    with pytest.raises(sprintstatus.ActionItemsMalformed, match="not a list"):
        sprintstatus.load_action_items(project_tree.sprint_status)


def test_action_items_malformed_is_a_sprint_status_error():
    # one `except SprintStatusError` still catches every reader fault
    assert issubclass(sprintstatus.ActionItemsMalformed, sprintstatus.SprintStatusError)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"action_items: [unclosed", "not valid YAML"),
        (b"- a\n- b\n", "no top-level mapping"),
        (b"", "no top-level mapping"),
        (b'action_items:\n  - id: "bad \xff byte"\n', "could not be read"),
    ],
    ids=["bad-yaml", "top-level-list", "empty-file", "undecodable"],
)
def test_load_action_items_unreadable_board_raises(project_tree, raw, match):
    """Every board-level fault is a typed `SprintStatusError` — including an
    undecodable file, which `read_text` raises as `UnicodeDecodeError` (a
    `ValueError`, not an `OSError`).

    Ablation: drop `UnicodeDecodeError` from the read's except tuple and the
    undecodable row escapes untyped."""
    project_tree.sprint_status.write_bytes(raw)
    with pytest.raises(sprintstatus.SprintStatusError, match=match) as exc_info:
        sprintstatus.load_action_items(project_tree.sprint_status)
    assert not isinstance(exc_info.value, sprintstatus.ActionItemsMalformed)


def test_load_is_unaffected_by_action_items(project_tree):
    """`load()` stays byte-for-byte behaviourally identical: the retro list is
    neither a key it classifies nor a shape it validates — a malformed one
    included, which only `load_action_items` refuses."""
    write_sprint(project_tree, {"epic-1": "in-progress", "1-1-user-auth": "done"})
    baseline = sprintstatus.load(project_tree.sprint_status)
    text = project_tree.sprint_status.read_text(encoding="utf-8")
    for tail in (_BOARD_WITH_ITEMS.split("\naction_items:", 1)[1], " not-a-list\n"):
        project_tree.sprint_status.write_text(text + "action_items:" + tail, encoding="utf-8")
        ss = sprintstatus.load(project_tree.sprint_status)
        assert (ss.epics, ss.stories, ss.retros, ss.retro_items, ss.unknown_keys) == (
            baseline.epics,
            baseline.stories,
            baseline.retros,
            baseline.retro_items,
            baseline.unknown_keys,
        )


def test_load_action_items_non_regular_board_raises(project_tree):
    """A board path that exists but is not a file is a fault, not absence.

    Ablation: restore `return None` for the non-`S_ISREG` arm and this answers
    None — indistinguishable from "no board"."""
    project_tree.sprint_status.unlink(missing_ok=True)
    project_tree.sprint_status.mkdir()
    with pytest.raises(sprintstatus.SprintStatusError, match="not a regular file"):
        sprintstatus.load_action_items(project_tree.sprint_status)


def test_load_action_items_metadata_fault_raises_typed(project_tree, monkeypatch):
    """A `stat` refusal is a `SprintStatusError` (not the malformed-list
    subclass), never a bare OSError and never "no board".

    Ablation: drop the `except OSError` arm around `stat()` and the
    PermissionError escapes untyped."""
    project_tree.sprint_status.write_text("action_items: []\n", encoding="utf-8")
    fault_metadata_probe(monkeypatch, project_tree.sprint_status, "stat")
    with pytest.raises(sprintstatus.SprintStatusError, match="could not be read") as exc_info:
        sprintstatus.load_action_items(project_tree.sprint_status)
    assert not isinstance(exc_info.value, sprintstatus.ActionItemsMalformed)


# ------------------------------------------------- value-span splice (DW-514/516)
#
# The one-line writer replaces exactly the value node's source span. Driven
# through `advanced_bytes`, the reproduction surface: each board below published
# invalid YAML or lost its comment when the edit rewrote everything after `key:`.


@pytest.mark.parametrize(
    ("board", "reason"),
    [
        # rewriting the anchored text would leave `*st` dangling (or silently
        # move every row that aliases it)
        (b"development_status:\n  1-1-a: &st backlog\n  1-2-b: *st\n", "value-is-anchored"),
        (b"development_status:\n  1-1-a: !!str &x backlog\n", "value-is-anchored"),
        # the row's line also carries the flow mapping's closing brace
        (b"development_status: {epic-1: in-progress,\n  1-1-a: backlog}\n", "mapping-is-flow"),
    ],
    ids=["anchored", "tag-then-anchor", "flow-mapping"],
)
def test_advance_refuses_anchored_and_flow_rows(tmp_path, board, reason):
    """Ablation: drop the refusals in `_locate_row` and the anchored boards publish
    a dangling alias or a dropped anchor, while the flow board falls back to
    `unreadable-value` (its closing brace fails the tail check), which the
    `reason` assertion catches."""
    with pytest.raises(sprintstatus.SprintStatusWriteRefused) as exc_info:
        sprintstatus.advanced_bytes(board, "1-1-a", "in-progress")
    assert exc_info.value.reason == reason

    p = tmp_path / "sprint-status.yaml"
    p.write_bytes(board)
    with pytest.raises(sprintstatus.SprintStatusWriteRefused) as exc_info:
        sprintstatus.advance(p, "1-1-a", "in-progress")
    assert exc_info.value.reason == reason
    assert p.read_bytes() == board  # nothing published


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (b"  1-1-a:  # note\n", b"  1-1-a: in-progress  # note\n"),
        (b"  1-1-a:\n", b"  1-1-a: in-progress\n"),
        (b'  1-1-a: "a # b"  # real comment\n', b"  1-1-a: in-progress  # real comment\n"),
        (b"  1-1-a: backlog  \n", b"  1-1-a: in-progress  \n"),
        (b"  1-1-a: backlog  # c\r\n", b"  1-1-a: in-progress  # c\r\n"),
        # a property-only value: the anchor scan must not reach the next row's key
        (b"  1-1-a: !!str\n  &k 1-3-c: backlog\n", b"  1-1-a: in-progress\n  &k 1-3-c: backlog\n"),
    ],
    ids=[
        "null-comment",
        "bare-null",
        "quoted-comment",
        "trailing-whitespace",
        "crlf-comment",
        "tag-only-before-anchored-key",
    ],
)
def test_advance_splices_only_the_value_span(row, expected):
    """Every byte outside the value span — gap, comment, trailing whitespace,
    terminator — survives as authored. Ablation: restore the old regex split and
    the null+comment row loses `# note` while the bare null is refused."""
    head = b"development_status:\n"
    tail = b"  1-2-b: backlog\n"

    out = sprintstatus.advanced_bytes(head + row + tail, "1-1-a", "in-progress")

    assert out == head + expected + tail
    assert sprintstatus.status_in_bytes(out, "1-1-a") == "in-progress"


# ------------------------------------------------------ reader key identity (DW-515)
#
# The writer resolves its row by the key `load()` reads — `str()` of the
# constructed key, epic/retro numbers as ints — not by the node's text. Two
# distinct constructed keys reading as one leave the reader's answer to
# iteration order, so the write path refuses them.


@pytest.mark.parametrize(
    ("board", "story_key", "refused_key", "current"),
    [
        # a str key and a date key, both `str()` to 2001-01-04: the reader reads
        # the first (backlog), the old writer rewrote the date row (done)
        (
            b'development_status:\n  "2001-01-04": backlog\n  2001-01-04: done\n',
            "2001-01-04",
            "2001-01-04",
            "backlog",
        ),
        # epic-1 and epic-01 are both epic 1: the reader's lift sees backlog
        # (last wins), the old writer regressed the `epic-1: done` row
        (
            b"development_status:\n  epic-1: done\n  epic-01: backlog\n  1-1-a: backlog\n",
            "1-1-a",
            "epic-1",
            "backlog",
        ),
        # the same collision as the first row, one key reached through a merge
        (
            b'development_status:\n  <<: {"2001-01-04": backlog}\n  2001-01-04: done\n',
            "2001-01-04",
            "2001-01-04",
            "backlog",
        ),
    ],
    ids=["str-date-collision", "epic-collision", "merge-explicit-collision"],
)
def test_advance_refuses_keys_the_reader_cannot_tell_apart(
    tmp_path, board, story_key, refused_key, current
):
    """Ablation: resolve with `_last_entry(mapping, key)` again and the
    str/date and merge boards publish an edit of the `done` row, while the epic
    board regresses `epic-1: done` to in-progress — no raise anywhere."""
    with pytest.raises(sprintstatus.SprintStatusWriteRefused) as exc_info:
        sprintstatus.advanced_bytes(board, story_key, "in-progress")
    err = exc_info.value
    assert (err.reason, err.story_key, err.current, err.target) == (
        "key-ambiguous",
        refused_key,
        current,
        "in-progress",
    )
    assert err.path is None

    p = tmp_path / "sprint-status.yaml"
    p.write_bytes(board)
    with pytest.raises(sprintstatus.SprintStatusWriteRefused) as exc_info:
        sprintstatus.advance(p, story_key, "in-progress", now="02-06-2026 09:00")
    err = exc_info.value
    assert (err.reason, err.story_key, err.current, err.path) == (
        "key-ambiguous",
        refused_key,
        current,
        p,
    )
    assert "keep exactly one" in str(err)
    assert p.read_bytes() == board  # nothing published: no story row, epic, or last_updated


@pytest.mark.parametrize(
    ("board", "expected", "epics"),
    [
        # a lone non-canonical epic key is still epic 1 to the reader: lift it
        # under its own spelling (the old writer looked for `epic-1:` and skipped)
        (
            b"development_status:\n  epic-01: backlog\n  1-1-a: backlog\n",
            b"development_status:\n  epic-01: in-progress\n  1-1-a: in-progress\n",
            {1: "in-progress"},
        ),
        # an exact duplicate constructs to one key: last wins, not ambiguous
        (
            b"development_status:\n  1-1-a: backlog\n  1-1-a: backlog\n",
            b"development_status:\n  1-1-a: backlog\n  1-1-a: in-progress\n",
            {},
        ),
        # the folded collapse rewrites the row under its own key text too
        (
            b"development_status:\n  epic-01:\n    backlog\n  1-1-a: backlog\n",
            b"development_status:\n  epic-01: in-progress\n  1-1-a: in-progress\n",
            {1: "in-progress"},
        ),
    ],
    ids=["non-canonical-lone-epic", "exact-duplicate", "folded-non-canonical-epic"],
)
def test_advance_edits_the_row_the_reader_reads(tmp_path, board, expected, epics):
    """Ablation: match `key` instead of `row.key_text` in `_set_mapping_value`
    and the lone `epic-01` row refuses `key-not-plain`, which the lift swallows,
    so it stays `backlog`; resolve with `_last_entry` and it is never found.
    Collapse the folded row under `{key}` instead of `{row.key_text}` and the
    folded board publishes `epic-1: in-progress`."""
    assert sprintstatus.advanced_bytes(board, "1-1-a", "in-progress") == expected

    p = tmp_path / "sprint-status.yaml"
    p.write_bytes(board)
    assert sprintstatus.advance(p, "1-1-a", "in-progress") == "in-progress"
    assert p.read_bytes() == expected
    assert sprintstatus.story_status(p, "1-1-a") == "in-progress"
    assert sprintstatus.load(p).epics == epics  # the reader sees the lifted row


def test_other_epic_lift_refusals_stay_best_effort(tmp_path):
    """Only `key-ambiguous` makes the lift loud: a quoted epic key the writer
    declines (`key-not-plain`) is left as authored and the story still moves.

    Ablation: raise on any `_Refused` epic edit and this reddens."""
    board = b'development_status:\n  "epic-1": backlog\n  1-1-a: backlog\n'
    p = tmp_path / "sprint-status.yaml"
    p.write_bytes(board)
    assert sprintstatus.advance(p, "1-1-a", "in-progress") == "in-progress"
    assert p.read_bytes() == b'development_status:\n  "epic-1": backlog\n  1-1-a: in-progress\n'
