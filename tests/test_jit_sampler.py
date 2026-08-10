"""Gate-1 tests for the numba batch sampler (Phase 2.6).

The batch sampler is NOT bitwise-compatible with the Python sampler: it is
seeded once per decision from the caller's Generator and then draws from
numba's internal RNG. That is by design (see kernel.sample_arrangements).
So these tests pin the properties that actually matter: legality,
determinism, statistical faithfulness, and that OPENHEARTS_NO_JIT=1 still
runs the untouched Python path bitwise.
"""
import os

import numpy as np
import pytest

from openhearts.belief.table import BeliefTable, Level
from openhearts.engine import cards, kernel
from openhearts.engine.game import deal
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.sampler.sampler import sample_arrangement
from openhearts.search.decision import SearchPlayer

pytestmark = pytest.mark.skipif(not kernel.HAVE_NUMBA,
                                reason="numba not installed")


def _tables(n_tables, seed0=0):
    """Diverse mid-game belief tables, taken only from PlayerViews."""
    out = []
    h = HeuristicPlayer()
    g = 0
    while len(out) < n_tables:
        state = deal(np.random.default_rng(seed0 + g))
        g += 1
        step = 0
        while not state.is_over() and len(out) < n_tables:
            seat = state.to_play
            view = state.view_for(seat)
            if 4 <= step <= 44 and step % 3 == 0:
                level = [Level.UNIFORM, Level.VOIDS,
                         Level.FULL][(step // 3) % 3]
                out.append(BeliefTable.from_view(view, level))
            state.play(h.choose(view))
            step += 1
    return out


# --------------------------------------------------------------- (a) legality
def test_batch_arrangements_are_legal_and_complete():
    rng = np.random.default_rng(0)
    tables = _tables(500)
    assert len(tables) == 500
    checked = 0
    for t in tables:
        hands_list, n_failed = kernel.sample_arrangements(t, rng, 3)
        assert n_failed + len(hands_list) == 3
        for hands in hands_list:
            union = 0
            for i, hnd in enumerate(hands):
                assert isinstance(hnd, int)
                assert bin(hnd).count("1") == t.hand_sizes[i]
                for s in t.voids[i]:
                    assert hnd & cards.SUIT_MASK[s] == 0
                # a zero in the table is permanent: never assign such a card
                for c in cards.cards_in(hnd):
                    assert t.probs[i, c] > 0.0
                assert union & hnd == 0
                union |= hnd
            assert union == t.unseen_mask
            checked += 1
    assert checked > 1000


# ----------------------------------------------------------- (b) determinism
def test_batch_is_deterministic_in_caller_rng_state():
    t = _tables(1, seed0=11)[0]
    a, fa = kernel.sample_arrangements(t, np.random.default_rng(7), 20)
    b, fb = kernel.sample_arrangements(t, np.random.default_rng(7), 20)
    assert (a, fa) == (b, fb)
    c, _ = kernel.sample_arrangements(t, np.random.default_rng(8), 20)
    assert c != a

    # and the caller's Generator is advanced, so successive calls differ
    rng = np.random.default_rng(7)
    d, _ = kernel.sample_arrangements(t, rng, 20)
    e, _ = kernel.sample_arrangements(t, rng, 20)
    assert d == a and e != d


# ------------------------------------------------- (c) statistical faithfulness
def _marginals(hands_list, n_cards=52):
    counts = np.zeros((3, n_cards))
    for hands in hands_list:
        for i, h in enumerate(hands):
            for c in cards.cards_in(h):
                counts[i, c] += 1
    return counts


def test_symmetric_table_marginals_match_probs():
    # Opening table: no voids, hand sizes 13/13/13, so the card-by-card
    # sampler's known bias vanishes by exchangeability and the empirical
    # marginals must match table.probs itself.
    state = deal(np.random.default_rng(3))
    t = BeliefTable.from_view(state.view_for(state.to_play), Level.FULL)
    assert t.hand_sizes == [13, 13, 13] and not any(t.voids)
    n = 20_000
    hands_list, n_failed = kernel.sample_arrangements(
        t, np.random.default_rng(123), n)
    assert n_failed == 0
    counts = _marginals(hands_list)
    unseen = cards.cards_in(t.unseen_mask)
    tested = 0
    for c in range(52):
        for i in range(3):
            p = t.probs[i, c]
            if p == 0.0:
                assert counts[i, c] == 0, "zero-probability cell was assigned"
                continue
            if n * p < 10 or n * (1 - p) < 10:
                continue
            sigma = np.sqrt(n * p * (1 - p))
            assert abs(counts[i, c] - n * p) <= 3 * sigma, (
                f"cell ({i},{c}) p={p:.4f} obs={counts[i, c]} exp={n*p:.1f}")
            tested += 1
    # every unseen cell is testable at this table (p ~ 1/3, N = 20k), so a
    # regression that silently skips cells cannot hide behind the filter
    assert tested == 3 * len(unseen)


def test_midgame_marginals_match_python_sampler():
    # Rebalanced mid-game table: here the card-by-card algorithm is knowingly
    # biased relative to table.probs, so the reference is the PYTHON sampler's
    # own empirical marginals. This tests port faithfulness, not the algorithm.
    state = deal(np.random.default_rng(4))
    h = HeuristicPlayer()
    for _ in range(30):
        state.play(h.choose(state.view_for(state.to_play)))
    t = BeliefTable.from_view(state.view_for(state.to_play), Level.FULL)
    assert any(t.voids), "want a table with real void constraints"

    n = 20_000
    jit_hands, jf = kernel.sample_arrangements(t, np.random.default_rng(5), n)
    assert jf == 0
    py_rng = np.random.default_rng(6)
    py_hands = []
    for _ in range(n):
        r = sample_arrangement(t, py_rng)
        assert r is not None
        py_hands.append(r[0])

    cj = _marginals(jit_hands) / n
    cp = _marginals(py_hands) / n
    tested = 0
    for c in range(52):
        for i in range(3):
            if t.probs[i, c] == 0.0:
                assert cj[i, c] == 0 and cp[i, c] == 0
                continue
            p = cp[i, c]
            if n * p < 10 or n * (1 - p) < 10:
                continue
            sigma = np.sqrt(p * (1 - p) * 2 / n)
            assert abs(cj[i, c] - cp[i, c]) <= 3 * sigma, (
                f"cell ({i},{c}) jit={cj[i, c]:.4f} py={cp[i, c]:.4f}")
            tested += 1
    assert tested >= 40, tested  # 45 non-zero cells at this table


# ------------------------------------------------------ (d) NO_JIT unchanged
def test_no_jit_forces_python_sampler_bitwise(monkeypatch):
    state = deal(np.random.default_rng(9))
    h = HeuristicPlayer()
    for _ in range(8):
        state.play(h.choose(state.view_for(state.to_play)))
    view = state.view_for(state.to_play)

    monkeypatch.setenv("OPENHEARTS_NO_JIT", "1")
    kernel.reset_jit_enabled()
    try:
        r1 = np.random.default_rng(21)
        a = SearchPlayer(Level.FULL, 30, r1, jit_sampler=True).choose(view)
        r2 = np.random.default_rng(21)
        b = SearchPlayer(Level.FULL, 30, r2, jit_sampler=False).choose(view)
        assert a == b
        # identical rng consumption too, not just the same answer
        assert r1.bit_generator.state == r2.bit_generator.state
    finally:
        monkeypatch.delenv("OPENHEARTS_NO_JIT", raising=False)
        kernel.reset_jit_enabled()
    assert os.environ.get("OPENHEARTS_NO_JIT") is None


def test_jit_sampler_default_off_for_searchplayer_and_on_for_honest():
    from openhearts.search.honest import HonestSearchPlayer
    assert SearchPlayer(Level.FULL, 4, np.random.default_rng(0)) \
        .jit_sampler is False
    hp = HonestSearchPlayer(Level.FULL, n_outer=4, n_inner=3,
                            rng=np.random.default_rng(0))
    assert hp.jit_sampler is True
    assert hp._inner.jit_sampler is True
