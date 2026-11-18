"""Registry-scoped teardown for the live psmux gate's probe sessions.

Lives outside ``test_psmux_live.py`` because that module is skipped wholesale
off Windows, and the teardown's timing policy has to be provable everywhere:
``tests/test_psmux_teardown.py`` drives it on every OS against a fake registry
whose servers are real child processes, on a fake clock. Only the process-table
witness (``_kill_unregistered_servers``, which shells out to PowerShell) stays
in the live module; it arrives here as the injected ``kill_unregistered``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from bmad_loop.adapters.tmux_base import TmuxError

# psmux's own client-side readiness deadline: `src/main.rs`, source-read at
# v3.3.8 — `ready_deadline = Instant::now() + Duration::from_secs(15)`, after which
# the client prints `psmux: failed to create session` and exits 1 WITHOUT killing
# the server it spawned. So it is also the longest a server may take to register
# while psmux still considers that a normal start.
PSMUX_READY_DEADLINE_S = 15.0

# How long absence must hold for a session that was seen: its port file existed,
# so both kill verbs could address it, and a short confirmation is honest.
SEEN_CONFIRM_S = 1.0

# The teardown's overall budget past the readiness deadline before it declares
# a leak rather than returning.
TEARDOWN_SLACK_S = 45.0

POLL_S = 0.5


def plain_has_session(mux: Any, session: str, *, env: dict[str, str] | None = None) -> bool:
    return mux._run(["has-session", "-t", session], check=False, env=env).returncode == 0


def seen_anywhere(mux: Any, session: str, env: dict[str, str]) -> bool:
    """True if the session answers in the isolated registry or in the default one."""
    return plain_has_session(mux, session, env=env) or plain_has_session(mux, session)


def teardown_probe_session(
    mux: Any,
    session: str,
    env: dict[str, str],
    *,
    known_created: bool,
    kill_unregistered: Callable[[str], list[str]],
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Kill a probe session and its server, and refuse to return until it is
    provably gone in BOTH the isolated and the default registry.

    RETRIED, and aimed at the REGISTRY rather than only at the name.

    `psmux: failed to create session` is the CLIENT's readiness poll timing out,
    not a creation failure (`src/main.rs`, source-read at v3.3.8: the message is
    printed once `ready_deadline` passes), so under load the server routinely
    comes up a moment after the mint reported failure. A single-shot
    `kill-session` fires while that server is still starting, misses, and both
    reads then answer "not there" — a clean-looking teardown over a real leak.
    Every leak makes the NEXT run's mint slower and its own timeout likelier,
    which is how one instrument failure cascades across a box (observed: a full
    suite going from 0 to 13 fixture errors as leaked servers accumulated).

    `kill-server` is what makes this decisive: it force-kills every server whose
    port file is under `psmux_dir()` (`src/main.rs`, source-read — it `read_dir`s
    that root), and every root here is a private temp directory holding nothing
    but the probe session. So it does not depend on the session having registered
    under its NAME yet, which is exactly what a mid-start server has not done.
    Both verbs are issued each pass because they fail in opposite directions: the
    name-scoped one works before the port file settles, the registry-scoped one
    after.

    The default-registry read is checked too: a build ignoring `PSMUX_DATA_DIR`
    would have created the session in the developer's real registry, and that is
    the one leak nothing here would otherwise catch.

    HOW LONG ABSENCE HAS TO HOLD depends on whether the session was ever THERE,
    and that asymmetry is the whole of the timing here.

    - Seen present: the server registered, so both verbs can address it and a
      short confirmation is honest — the port file is gone and stays gone.
    - Never seen: the server may simply not have registered YET, and a mid-start
      server is indistinguishable from no server at all. Both `has-session` and
      `kill-server` work off the port files under the root, so neither can reach
      one that has not written its own. Two absent reads a beat apart mean nothing
      here — measured: a delayed registration let an earlier revision return after
      0.50s with the server visible immediately afterwards.

    "Seen" is either this teardown's own read or ``known_created``: the caller's
    POSITIVE observation of the session after its mint (a `has-session` that
    answered). That is the same fact the first read here would record, taken
    earlier — which matters for a test that kills its own session in the body
    (`test_adopted_kill_session_honors_the_exact_match_target`): by teardown the
    session is already gone, the first read sees nothing, and without the carried
    fact the teardown held the full never-seen vigil and asked the process table
    over a session that had registered and been killed. Only a positive read may
    set it. A mint that failed, raised, or was never confirmed passes False and
    keeps the whole vigil below.

    So the unseen case holds its vigil for `PSMUX_READY_DEADLINE_S`, which is not
    a guessed number: it is the CLIENT's own readiness deadline (`src/main.rs`,
    source-read at v3.3.8 — `ready_deadline = Instant::now() + 15s`, then
    `psmux: failed to create session` and `exit(1)`). A client that gave up there
    does NOT take the server down with it, so 15s is exactly how long psmux itself
    is prepared to wait for a registration, and the kills keep firing throughout —
    the moment a port file appears, `kill-server` reaches it.

    Only the pathological path pays that: every fixture here tears down a session
    it minted and observed, so either the first read sees it or the carried fact
    stands in for it, and teardown costs a beat.

    A first read that RAISES (a timed-out or failed `has-session`, most likely
    right after an overloaded mint — the path a mid-start server leaks on) is no
    observation at all: it must not abort the teardown before either kill verb
    fires, and it is not a positive read, so it leaves ``seen`` at the carried
    fact and the loop below takes over.
    """
    try:
        seen = known_created or seen_anywhere(mux, session, env)
    except (OSError, TmuxError, subprocess.TimeoutExpired):
        seen = known_created
    deadline = clock() + PSMUX_READY_DEADLINE_S + TEARDOWN_SLACK_S
    quiet_since: float | None = None
    while clock() < deadline:
        try:
            mux._run(["kill-session", "-t", session], check=False, env=env)
            mux._run(["kill-server"], check=False, env=env)
            present = seen_anywhere(mux, session, env)
        except (OSError, TmuxError, subprocess.TimeoutExpired):
            present = True
        if present:
            seen = True  # it registered after all; the kills can address it now
            quiet_since = None
        else:
            needed = SEEN_CONFIRM_S if seen else PSMUX_READY_DEADLINE_S
            now = clock()
            if quiet_since is None:
                quiet_since = now
            elif now - quiet_since >= needed:
                if seen:
                    return  # it was addressable, the kill landed, it is gone
                # Never seen, and no psmux verb can see it now — which is also
                # true of a server running under a root it could not write. Ask
                # the process table before calling this death.
                killed = kill_unregistered(session)
                if not killed:
                    return
                print(
                    f"warning: probe session {session} was running with an "
                    f"unwritable registry — no psmux verb could reach it; killed "
                    f"pid(s) {', '.join(killed)} directly",
                    file=sys.stderr,
                )
                quiet_since = None  # re-confirm now that something was killed
        sleep(POLL_S)
    raise AssertionError(
        f"probe setup: probe session {session} survived teardown; kill it manually"
    )


@contextmanager
def minted_session(mint: Callable[[], None], teardown: Callable[[bool], None]) -> Iterator[None]:
    """Run ``mint``, yield, and always call ``teardown(known_created)``.

    ``known_created`` is True only once ``mint`` has RETURNED — callers make it
    return only after a positive observation of the session — so a mint that
    raised (the client's readiness timeout, a failed has-session) tears down with
    the full never-seen vigil, while a body that killed its own session keeps
    the fact it was once there.
    """
    created = False
    try:
        mint()
        created = True
        yield
    finally:
        teardown(created)
