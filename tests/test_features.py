"""Phase 4 Task 1: the position featurizer's forever-contract.

Three gates from the plan: rotation invariance, determinism + [0,1] bounds,
and consistency across >=1,000 random mid-game states. Plus the house
equivalence gate: the compiled kernel and the Python fallback must agree
IN THE SAME PROCESS (running the suite twice under two env settings never
compares the two paths to each other).

The relabelled positions in the rotation test are built inline with explicit
indices on purpose -- reusing the module's own rotation helper would let a
direction flip pass vacuously.
"""
import numpy as np
import pytest

from openhearts.engine import cards, features, kernel
from openhearts.engine.game import deal
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.players.random_player import RandomPlayer


# ------------------------------------------------------------------ helpers
def _arrays(state):
    """kernel-style array representation of a GameState (played incl. trick)."""
    hands = np.array(state.hands, dtype=np.int64)
    scores = np.array(state.scores, dtype=np.int64)
    trick_cards = np.zeros(4, dtype=np.int64)
    trick_seats = np.zeros(4, dtype=np.int64)
    played = 0
    for _s, c in state.history:
        played |= 1 << c
    for i, (s, c) in enumerate(state.current_trick):
        trick_cards[i] = c
        trick_seats[i] = s
        played |= 1 << c
    tl = len(state.current_trick)
    if tl:
        led = int(trick_cards[0]) // 13
        win_rank = int(trick_cards[0]) % 13
        win_seat = int(trick_seats[0])
        for i in range(1, tl):
            c = int(trick_cards[i])
            if c // 13 == led and c % 13 > win_rank:
                win_rank, win_seat = c % 13, int(trick_seats[i])
    else:
        led, win_seat = -1, -1
    return dict(hands=hands, played_mask=np.int64(played),
                trick_cards=trick_cards, trick_seats=trick_seats,
                trick_len=tl, led_suit=led, win_seat=win_seat,
                hearts_broken=state.hearts_broken,
                trick_number=state.trick_number, scores=scores)


def _call(pos, seat, fn=None):
    fn = fn or features.featurize
    return fn(pos["hands"], pos["played_mask"], pos["trick_cards"],
              pos["trick_seats"], pos["trick_len"], pos["led_suit"],
              pos["win_seat"], pos["hearts_broken"], pos["trick_number"],
              pos["scores"], seat)


def _positions(n_games, seed, mix=True):
    """Mid-game positions from heuristic and random prefixes."""
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_games):
        state = deal(rng)
        players = [(HeuristicPlayer() if (not mix or (g + i) % 2 == 0)
                    else RandomPlayer(rng)) for i in range(4)]
        while not state.is_over():
            out.append(_arrays(state))
            s = state.to_play
            state.play(players[s].choose(state.view_for(s)))
        out.append(_arrays(state))  # terminal position
    return out


def _popcount(m):
    return bin(int(m) & ((1 << 52) - 1)).count("1")


# ------------------------------------------------------------------- layout
def test_layout_constants():
    assert features.FEATURES_V == 1
    assert features.NF == 333
    assert (features.OFF_HANDS, features.OFF_PLAYED, features.OFF_TRICK,
            features.OFF_WIN_SEAT, features.OFF_LED, features.OFF_SCALARS) \
        == (0, 208, 260, 312, 316, 320)


# ------------------------------------------------------- gate 1: rotation
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rotation_invariance(seed):
    for pos in _positions(2, seed):
        for k in range(4):
            hands = np.array([pos["hands"][(k + r) % 4] for r in range(4)],
                             dtype=np.int64)
            scores = np.array([pos["scores"][(k + r) % 4] for r in range(4)],
                              dtype=np.int64)
            tseats = np.zeros(4, dtype=np.int64)
            for i in range(pos["trick_len"]):
                tseats[i] = (int(pos["trick_seats"][i]) - k) % 4
            ws = pos["win_seat"]
            ws = -1 if ws < 0 else (int(ws) - k) % 4
            rel = dict(pos, hands=hands, scores=scores, trick_seats=tseats,
                       win_seat=ws)
            assert np.array_equal(_call(pos, k), _call(rel, 0))


# --------------------------------------- gate 2: determinism + [0,1] bounds
def test_determinism_and_bounds():
    for pos in _positions(6, 11):
        for seat in range(4):
            a = _call(pos, seat)
            b = _call(pos, seat)
            assert a.dtype == np.float64 and a.shape == (features.NF,)
            assert np.array_equal(a, b)
            assert np.all(a >= 0.0) and np.all(a <= 1.0)


# ------------------------------------------------- gate 3: consistency @1k+
def test_consistency_over_many_states():
    positions = _positions(40, 99)
    assert len(positions) >= 1000
    for pos in positions:
        seat = int(pos["trick_number"] + pos["trick_len"]) % 4
        f = _call(pos, seat)
        played = _popcount(pos["played_mask"])
        hand_block = f[features.OFF_HANDS:features.OFF_PLAYED]
        played_block = f[features.OFF_PLAYED:features.OFF_TRICK]
        trick_block = f[features.OFF_TRICK:features.OFF_WIN_SEAT]
        assert hand_block.sum() == 52 - played
        assert played_block.sum() == played
        assert played == 4 * pos["trick_number"] + pos["trick_len"]
        assert trick_block.sum() == pos["trick_len"]
        # a card is either still held or already played, never both
        held_any = hand_block.reshape(4, 52).sum(axis=0)
        assert np.all(held_any <= 1.0)
        assert not np.any((held_any > 0) & (played_block > 0))
        # trick cards are a subset of played cards
        assert not np.any((trick_block > 0) & (played_block == 0))
        assert f[features.OFF_SCALARS] == pos["trick_number"] / 13.0
        assert f[features.OFF_SCALARS + 1] == pos["trick_len"] / 4.0
        assert f[features.OFF_SCALARS + 2] == float(pos["hearts_broken"])
        assert f[features.OFF_SCALARS + 8] == float(
            (int(pos["played_mask"]) >> cards.QUEEN_SPADES) & 1)
        if pos["trick_len"] == 0:
            assert f[features.OFF_LED:features.OFF_SCALARS].sum() == 0.0
            assert f[features.OFF_WIN_SEAT:features.OFF_LED].sum() == 0.0
        else:
            assert f[features.OFF_LED + pos["led_suit"]] == 1.0
            assert f[features.OFF_LED:features.OFF_SCALARS].sum() == 1.0
            assert f[features.OFF_WIN_SEAT:features.OFF_LED].sum() == 1.0
            rot = (int(pos["win_seat"]) - seat) % 4
            assert f[features.OFF_WIN_SEAT + rot] == 1.0


def test_terminal_and_opening_positions():
    """All-empty and trick-0 positions must not index with led_suit == -1."""
    rng = np.random.default_rng(5)
    state = deal(rng)
    opening = _arrays(state)
    f = _call(opening, state.to_play)
    assert f[features.OFF_HANDS:features.OFF_PLAYED].sum() == 52
    assert f[features.OFF_PLAYED:features.OFF_TRICK].sum() == 0

    players = [HeuristicPlayer() for _ in range(4)]
    while not state.is_over():
        state.play(players[state.to_play].choose(
            state.view_for(state.to_play)))
    term = _arrays(state)
    g = _call(term, 0)
    assert g[features.OFF_HANDS:features.OFF_PLAYED].sum() == 0
    assert g[features.OFF_PLAYED:features.OFF_TRICK].sum() == 52
    assert np.all(g >= 0.0) and np.all(g <= 1.0)
    assert g[features.OFF_SCALARS] == 1.0
    assert sum(state.scores) == 26


# ------------------------------------------- house gate: JIT == Python path
@pytest.mark.skipif(not kernel.HAVE_NUMBA, reason="numba not installed")
def test_jit_matches_python_in_same_process():
    for pos in _positions(4, 7):
        for seat in range(4):
            a = _call(pos, seat, features._featurize_njit)
            b = _call(pos, seat, features._featurize_py)
            assert np.array_equal(a, b)
