"""The numba playout kernel must be indistinguishable from the Python one.

A silent divergence here would corrupt every downstream experiment without
failing anything, so these tests compare the full play SEQUENCE (not just the
final scores) across thousands of generated states, plus the specific
hand shapes that random generation under-samples.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from openhearts.belief.table import Level
from openhearts.engine import cards, kernel
from openhearts.engine.game import deal, legal_moves
from openhearts.engine.state import GameState
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.players.random_player import RandomPlayer
from openhearts.search.decision import SearchPlayer
from openhearts.search.honest import HonestSearchPlayer

pytestmark = pytest.mark.skipif(not kernel.HAVE_NUMBA,
                                reason="numba not installed")

HEURISTIC = HeuristicPlayer()


# ---------------------------------------------------------------- references
def python_playout(state, record):
    """The pure-Python playout, recording every (seat, card)."""
    while not state.is_over():
        seat = state.to_play
        card = HEURISTIC.choose(state.view_for(seat))
        record.append((seat, card))
        state.play(card)


def python_until_decision(state, stop_seat, record):
    """Python twin of playout_until_decision (honest.py's interception)."""
    while not state.is_over():
        seat = state.to_play
        view = state.view_for(seat)
        if seat == stop_seat and len(cards.cards_in(view.legal_moves)) > 1:
            return True
        card = HEURISTIC.choose(view)
        record.append((seat, card))
        state.play(card)
    return False


def kernel_playout(state, record):
    before = len(state.history) + len(state.current_trick)
    kernel.run_playout(state)
    record.extend((state.history + state.current_trick)[before:])


def kernel_until_decision(state, stop_seat, record):
    before = len(state.history) + len(state.current_trick)
    stopped = kernel.run_playout_until_decision(state, stop_seat)
    record.extend((state.history + state.current_trick)[before:])
    return stopped


def state_fingerprint(state):
    return (tuple(state.hands), tuple(state.scores), state.to_play,
            state.hearts_broken, state.trick_number,
            tuple(state.history), tuple(state.current_trick))


# ---------------------------------------------------------------- generators
def generate_states(n, seed=0):
    """Random-length prefixes of games played by mixed random/heuristic seats.

    Includes mid-trick states (prefix length is in plies, not tricks) and
    every phase from a fresh deal to the last trick.
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        state = deal(np.random.default_rng(int(rng.integers(1 << 30))))
        players = [HEURISTIC if rng.random() < 0.5
                   else RandomPlayer(rng) for _ in range(4)]
        plies = int(rng.integers(0, 46))
        for _ in range(plies):
            if state.is_over():
                break
            seat = state.to_play
            state.play(players[seat].choose(state.view_for(seat)))
        if state.is_over():
            continue
        out.append(state)
    return out


# ---------------------------------------------------------------- main gates
def test_playout_sequences_identical_over_many_states():
    states = generate_states(5200, seed=11)
    mid_trick = sum(1 for s in states if s.current_trick)
    late = sum(1 for s in states if s.trick_number >= 8)
    assert len(states) >= 5000
    assert mid_trick > 500 and late > 500, (mid_trick, late)
    for i, state in enumerate(states):
        pa, pb = [], []
        a = state.copy()
        b = state.copy()
        python_playout(a, pa)
        kernel_playout(b, pb)
        assert pa == pb, f"play sequence diverged on state {i}"
        assert a.scores == b.scores, f"scores diverged on state {i}"
        assert state_fingerprint(a) == state_fingerprint(b), (
            f"final state diverged on state {i}")


def test_until_decision_stops_at_the_same_ply():
    states = generate_states(2000, seed=12)
    stopped_any = 0
    for i, state in enumerate(states):
        for stop_seat in range(4):
            pa, pb = [], []
            a = state.copy()
            b = state.copy()
            sa = python_until_decision(a, stop_seat, pa)
            sb = kernel_until_decision(b, stop_seat, pb)
            assert sa == sb, f"stop flag diverged, state {i} seat {stop_seat}"
            assert len(pa) == len(pb), (
                f"stopped at a different ply, state {i} seat {stop_seat}")
            assert pa == pb, f"prefix diverged, state {i} seat {stop_seat}"
            assert state_fingerprint(a) == state_fingerprint(b), (
                f"state diverged, state {i} seat {stop_seat}")
            stopped_any += sa
    assert stopped_any > 1000


def test_until_decision_does_not_stop_at_a_forced_move():
    """A forced move at our seat is played through, not intercepted."""
    checked = 0
    for state in generate_states(400, seed=13):
        for stop_seat in range(4):
            a = state.copy()
            stopped = kernel_until_decision(a, stop_seat, [])
            if not stopped:
                continue
            view = a.view_for(a.to_play)
            assert a.to_play == stop_seat
            assert len(cards.cards_in(view.legal_moves)) > 1
            checked += 1
    assert checked > 200


def test_scores_sum_to_26_from_fresh_deals():
    for seed in range(50):
        state = deal(np.random.default_rng(seed))
        kernel.run_playout(state)
        assert sum(state.scores) == 26
        assert state.is_over()
        assert len(state.history) == 52


# ------------------------------------------------- explicitly-built edge cases
def hand_of(names):
    """names like 'Qs' -> card ints, using cards.card_name's alphabet."""
    out = 0
    for n in names:
        for c in range(52):
            if cards.card_name(c) == n:
                out |= cards.bit(c)
                break
        else:  # pragma: no cover
            raise AssertionError(n)
    return out


def compare_from(state):
    pa, pb = [], []
    a, b = state.copy(), state.copy()
    python_playout(a, pa)
    kernel_playout(b, pb)
    assert pa == pb
    assert state_fingerprint(a) == state_fingerprint(b)
    return pa


def test_spade_avoid_leaves_a_single_suit():
    # leader holds Qs + one club; Ks/As unseen -> spades deleted from the
    # candidate suits even though that leaves exactly one suit to lead.
    hands = [0, 0, 0, 0]
    hands[0] = hand_of(["Qs", "2c", "3c"])
    hands[1] = hand_of(["4c", "5c", "4s"])
    hands[2] = hand_of(["6c", "7c", "5s"])
    hands[3] = hand_of(["8c", "9c", "6s"])
    state = GameState(hands=hands, trick_number=5, hearts_broken=True,
                      to_play=0)
    seq = compare_from(state)
    assert cards.card_name(seq[0][1]) == "2c"


def test_spade_avoid_not_applied_when_spades_is_the_only_suit():
    hands = [0, 0, 0, 0]
    hands[0] = hand_of(["Qs", "2s"])
    hands[1] = hand_of(["3s", "4s"])
    hands[2] = hand_of(["5s", "6s"])
    hands[3] = hand_of(["7s", "8s"])
    state = GameState(hands=hands, trick_number=5, hearts_broken=True,
                      to_play=0)
    seq = compare_from(state)
    assert cards.card_name(seq[0][1]) == "2s"


def test_mid_trick_with_heart_already_played_into_the_live_trick():
    hands = [0, 0, 0, 0]
    hands[0] = hand_of(["3h", "4h"])
    hands[1] = hand_of(["3c", "5h"])
    hands[2] = hand_of(["4c", "6h"])
    hands[3] = hand_of(["5c", "7h"])
    state = GameState(hands=hands, trick_number=5, to_play=0)
    state.play(cards.cards_in(hand_of(["3h"]))[0])
    assert state.hearts_broken and len(state.current_trick) == 1
    compare_from(state)


def test_mid_trick_high_spades_sitting_in_the_live_trick():
    # Ks/As of spades are in the current (incomplete) trick, so they are NOT
    # in `history` -- the leading rule must see them the same way both sides.
    hands = [0, 0, 0, 0]
    hands[0] = hand_of(["Ks", "2c", "3c"])
    hands[1] = hand_of(["As", "4c", "5c"])
    hands[2] = hand_of(["Qs", "6c", "7c"])
    hands[3] = hand_of(["2s", "8c", "9c"])
    state = GameState(hands=hands, trick_number=5, hearts_broken=True,
                      to_play=0)
    state.play(cards.cards_in(hand_of(["Ks"]))[0])
    state.play(cards.cards_in(hand_of(["As"]))[0])
    assert len(state.current_trick) == 2
    compare_from(state)


def test_honest_playout_chain_matches_python():
    """The production honest path is three segments chained.

    kernel-until-decision -> Python inner re-determinization -> kernel-to-end
    must reproduce the pure-Python interception loop exactly, including the
    mid-trick history/current_trick rebuilt between the segments (the inner
    SearchPlayer reads them through `view_for`).
    """
    for i, state in enumerate(generate_states(150, seed=17)):
        for our_seat in (0, 2):
            a, b = state.copy(), state.copy()
            pa = HonestSearchPlayer(Level.FULL, n_outer=4, n_inner=3,
                                    rng=np.random.default_rng(99))
            pb = HonestSearchPlayer(Level.FULL, n_outer=4, n_inner=3,
                                    rng=np.random.default_rng(99))
            pa._playout_jit(a, our_seat)
            pb._playout_python(b, our_seat)
            assert state_fingerprint(a) == state_fingerprint(b), (i, our_seat)
            assert a.is_over()


# ------------------------------------------------- gate 2: JIT vs NO_JIT
def _decisions_with(jit_on):
    prev = os.environ.get("OPENHEARTS_NO_JIT")
    os.environ["OPENHEARTS_NO_JIT"] = "0" if jit_on else "1"
    kernel.reset_jit_enabled()
    try:
        assert kernel.jit_enabled() == jit_on
        out = []
        for seed in range(14):
            state = deal(np.random.default_rng(100 + seed))
            # jit_sampler=False on both: this gate pins the PLAYOUT kernel
            # against its Python reference, which requires an identical rng
            # stream. The batch sampler (Phase 2.6) deliberately draws a
            # different stream, so it is held out here and pinned separately
            # in tests/test_jit_sampler.py.
            search = SearchPlayer(Level.FULL, 12,
                                  np.random.default_rng(seed),
                                  jit_sampler=False)
            honest = HonestSearchPlayer(Level.FULL, n_outer=8, n_inner=4,
                                        rng=np.random.default_rng(seed),
                                        jit_sampler=False)
            for _ in range(11):
                view = state.view_for(state.to_play)
                if len(cards.cards_in(view.legal_moves)) > 1:
                    out.append(search.choose(view))
                    out.append(honest.choose(view))
                state.play(HEURISTIC.choose(view))
        return out
    finally:
        if prev is None:
            os.environ.pop("OPENHEARTS_NO_JIT")
        else:
            os.environ["OPENHEARTS_NO_JIT"] = prev
        kernel.reset_jit_enabled()


def test_same_decisions_with_and_without_jit():
    on = _decisions_with(True)
    off = _decisions_with(False)
    assert len(on) >= 200, len(on)
    assert on == off


def test_no_jit_env_var_disables_the_kernel_in_a_fresh_process():
    code = ("from openhearts.engine import kernel;"
            "print(kernel.jit_enabled())")
    env = dict(os.environ, OPENHEARTS_NO_JIT="1")
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_legal_moves_port_matches_engine():
    rng = np.random.default_rng(3)
    for _ in range(3000):
        hand = int(rng.integers(1, 1 << 52))
        trick_number = int(rng.integers(0, 13))
        hearts_broken = bool(rng.integers(0, 2))
        trick_len = int(rng.integers(0, 4))
        led = int(rng.integers(0, 4))
        if trick_len:
            # only current_trick[0][1] is read by legal_moves, so the
            # follower cards are irrelevant filler here
            lead_card = led * 13 + int(rng.integers(0, 13))
            trick = tuple((i, lead_card) for i in range(trick_len))
        else:
            trick = ()
            led = -1
        expected = legal_moves(hand, trick, hearts_broken, trick_number)
        got = int(kernel._legal(np.int64(hand), led, trick_len,
                                hearts_broken, trick_number))
        assert got & ((1 << 52) - 1) == expected
