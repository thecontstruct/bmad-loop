"""The live psmux gate's teardown policy, proven on every OS.

``tests/test_psmux_live.py`` runs only on Windows with psmux installed, yet its
teardown decides whether a probe server can leak for the rest of a CI job. The
timing asymmetry it rests on — a short confirmation for a session that was
seen, the full client readiness deadline for one that never was — is driven
here against a fake registry: port files per registry root, psmux servers that
are REAL child processes (so "no owned process left" is a process-table fact,
not a flag), registration that can land late, and a fake clock that the
teardown's own sleeps advance. Nothing here launches psmux or spends a token.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field

import psmux_teardown
import pytest

from bmad_loop.adapters.tmux_base import TmuxError

PRIVATE = "private-registry"
DEFAULT = "default-registry"
ENV = {"PSMUX_DATA_DIR": PRIVATE}
SESSION = "bmad-loop-test-fake"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeServer:
    session: str
    root: str
    proc: subprocess.Popen[bytes]
    registers_at: float
    writes_port: bool

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None


@dataclass
class FakeRegistry:
    """psmux as the teardown sees it. `has-session` and both kill verbs resolve
    through registered port files under the call's root (`env=None` is the
    default root, as for an instance spawned with PSMUX_DATA_DIR unset);
    `kill_unregistered` is the process-table witness, which sees a server's
    session token whether or not it ever registered."""

    clock: FakeClock
    servers: list[FakeServer] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)
    witness_calls: list[str] = field(default_factory=list)
    witness_killed: list[str] = field(default_factory=list)
    # Exceptions the next psmux calls raise, one per call, before resolving.
    failures: list[BaseException] = field(default_factory=list)

    def spawn(
        self,
        session: str = SESSION,
        *,
        root: str = PRIVATE,
        registers_at: float = 0.0,
        writes_port: bool = True,
    ) -> FakeServer:
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(300)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        server = FakeServer(session, root, proc, registers_at, writes_port)
        self.servers.append(server)
        return server

    def _registered(self, server: FakeServer, root: str) -> bool:
        return (
            server.alive
            and server.writes_port
            and server.root == root
            and self.clock() >= server.registers_at
        )

    @staticmethod
    def _kill(server: FakeServer) -> None:
        server.proc.kill()
        server.proc.wait(timeout=30)

    def _run(
        self, argv: list[str], *, check: bool = False, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        del check
        root = DEFAULT if env is None else env.get("PSMUX_DATA_DIR", DEFAULT)
        verb = argv[0]
        self.calls.append((verb, root))
        if self.failures:
            raise self.failures.pop(0)
        if verb == "has-session":
            hit = [s for s in self.servers if s.session == argv[2] and self._registered(s, root)]
        elif verb == "kill-session":
            hit = [s for s in self.servers if s.session == argv[2] and self._registered(s, root)]
            for server in hit:
                self._kill(server)
        elif verb == "kill-server":
            hit = [s for s in self.servers if self._registered(s, root)]
            for server in hit:
                self._kill(server)
        else:
            raise AssertionError(f"unexpected psmux verb {argv!r}")
        return subprocess.CompletedProcess(argv, 0 if hit else 1, "", "")

    def kill_unregistered(self, session: str) -> list[str]:
        self.witness_calls.append(session)
        doomed = [s for s in self.servers if s.alive and s.session == session]
        for server in doomed:
            self._kill(server)
        killed = [str(server.proc.pid) for server in doomed]
        self.witness_killed.extend(killed)
        return killed

    def alive(self) -> list[FakeServer]:
        return [server for server in self.servers if server.alive]


@pytest.fixture
def registry():
    reg = FakeRegistry(FakeClock())
    try:
        yield reg
    finally:
        for server in reg.servers:  # reap whatever a failing row left standing
            if server.alive:
                server.proc.kill()
            server.proc.wait(timeout=30)


def _teardown(registry: FakeRegistry, *, known_created: bool) -> None:
    psmux_teardown.teardown_probe_session(
        registry,
        SESSION,
        ENV,
        known_created=known_created,
        kill_unregistered=registry.kill_unregistered,
        clock=registry.clock,
        sleep=registry.clock.sleep,
    )


def test_a_known_created_session_its_test_killed_confirms_briefly(registry):
    """The case the carried fact exists for: the fixture observed the session,
    the test body killed it, and teardown's own first read sees nothing. Known
    created, absence needs only the seen-session confirmation — no readiness
    vigil and no process-table sweep over a session that registered.

    Ablation: drop `known_created` from the teardown's `seen` and this row reads
    the full vigil and a witness call."""
    server = registry.spawn()
    assert psmux_teardown.plain_has_session(registry, SESSION, env=ENV), "fake setup"
    registry._run(["kill-session", "-t", SESSION], env=ENV)  # the test body's own kill
    assert not server.alive, "fake setup: the body's kill landed"

    _teardown(registry, known_created=True)

    assert registry.clock.now < psmux_teardown.PSMUX_READY_DEADLINE_S
    assert registry.clock.now <= psmux_teardown.SEEN_CONFIRM_S + 2 * psmux_teardown.POLL_S
    assert registry.witness_calls == []
    assert registry.alive() == []


def test_without_the_fact_the_same_dead_session_holds_the_vigil(registry):
    """The contrast that makes the row above mean something: the identical
    already-killed session, torn down with no creation fact, is indistinguishable
    from one still starting, so it holds the readiness deadline and asks the
    process table before calling it gone."""
    registry.spawn()
    registry._run(["kill-session", "-t", SESSION], env=ENV)

    _teardown(registry, known_created=False)

    assert registry.clock.now >= psmux_teardown.PSMUX_READY_DEADLINE_S
    assert registry.witness_calls == [SESSION]
    assert registry.alive() == []


@pytest.mark.parametrize("registers_at", [0.6, 5.0, 14.0])
def test_a_never_observed_server_registering_late_is_caught(registry, registers_at):
    """An uncertain mint — the client gave up, or was never confirmed — whose
    server registers while the teardown is already running. The teardown must
    not return before that registration, must kill the server once it is
    addressable, and must leave its process dead.

    Ablation: treat every session as seen (`seen = True`) and the later rows
    return after the one-second confirmation with the server still starting."""
    server = registry.spawn(registers_at=registers_at)

    _teardown(registry, known_created=False)

    assert registry.clock.now >= max(registers_at, psmux_teardown.PSMUX_READY_DEADLINE_S)
    assert not server.alive
    assert registry.alive() == []
    # the registry kills reached it the pass it registered; the process-table
    # witness, still consulted after the vigil, found nothing left to kill
    assert registry.witness_killed == []


def test_a_server_that_never_registers_is_found_in_the_process_table(registry, capsys):
    """A server running under a root it could not write publishes no port file,
    so no psmux verb can reach it. Never seen, the teardown asks the process
    table after the vigil, kills it there, re-confirms and says so. (A carried
    creation fact cannot describe this server: `minted_session` sets it only
    after a `has-session` answered, which needs the port file this one never
    wrote.)

    Ablation: skip the witness (`killed = []`) and the row returns over the
    live server."""
    server = registry.spawn(writes_port=False)

    _teardown(registry, known_created=False)

    assert registry.witness_killed == [str(server.proc.pid)]
    assert not server.alive
    assert registry.alive() == []
    assert "unwritable registry" in capsys.readouterr().err


@pytest.mark.parametrize("known_created", [True, False], ids=["known-created", "never-seen"])
def test_a_default_registry_leak_is_never_a_clean_teardown(registry, known_created):
    """A build ignoring PSMUX_DATA_DIR lands the session in the operator's real
    registry, which the private-root kills cannot address. The default-registry
    read keeps it present, so the teardown refuses to return clean whatever the
    creation fact says.

    Ablation: drop the env-less read from `seen_anywhere` and both rows return
    over the live server."""
    leaked = registry.spawn(root=DEFAULT)

    with pytest.raises(AssertionError, match="survived teardown"):
        _teardown(registry, known_created=known_created)

    assert leaked.alive, "the private kills never reached the default registry"
    assert (("has-session", DEFAULT)) in registry.calls


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(["psmux", "has-session"], 5),
        OSError("psmux vanished"),
        TmuxError("has-session failed"),
    ],
    ids=["timeout", "oserror", "tmuxerror"],
)
def test_a_failed_first_probe_still_runs_the_teardown(registry, failure):
    """The first `has-session` raising — likeliest right after an overloaded
    mint, exactly when a mid-start server is loose — must not abort the
    teardown before a kill fires. It is no positive read either, so the server
    that registers late is still waited for and killed.

    Ablation: let the first `seen_anywhere` raise and every row escapes with
    the server alive."""
    server = registry.spawn(registers_at=5.0)
    registry.failures.append(failure)

    _teardown(registry, known_created=False)

    assert not server.alive
    assert registry.alive() == []
    assert registry.clock.now >= psmux_teardown.PSMUX_READY_DEADLINE_S


def test_minted_session_carries_creation_only_after_the_mint_returns(registry):
    """The fixture wiring: `known_created` is True exactly when the mint returned
    (after its positive read), whatever the body then does to the session."""
    seen: list[bool] = []

    def failing_mint() -> None:
        registry.spawn(registers_at=12.0)  # the server outlives its client
        raise AssertionError("probe setup: probe session creation failed")

    with pytest.raises(AssertionError, match="creation failed"):
        with psmux_teardown.minted_session(
            failing_mint,
            lambda known: (seen.append(known), _teardown(registry, known_created=known)),
        ):
            raise AssertionError("unreachable: the body never runs after a failed mint")
    assert seen == [False]
    assert registry.clock.now >= 12.0  # the vigil outlasted the late registration
    assert registry.alive() == []

    def confirmed_mint() -> None:
        registry.spawn()
        assert psmux_teardown.plain_has_session(registry, SESSION, env=ENV)

    with pytest.raises(RuntimeError, match="body failed"):
        with psmux_teardown.minted_session(confirmed_mint, seen.append):
            raise RuntimeError("body failed")
    with psmux_teardown.minted_session(confirmed_mint, seen.append):
        registry._run(["kill-session", "-t", SESSION], env=ENV)
    assert seen == [False, True, True]
