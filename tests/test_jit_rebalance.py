"""Phase 2.7 gate 1: the compiled rebalance must be BITWISE identical.

`_rebalance` is deterministic (no rng), so "close enough" is not a standard
we accept here: every float64 bit of the returned table must match the Python
reference, on a corpus that actually reaches the awkward corners (the
boundary-zero / coarse-tolerance path described in `table._rebalance`'s
docstring, which shows up in tricks 8-13).
"""
import os
import subprocess
import sys

import numpy as np
import pytest

from openhearts.belief import table as T
from openhearts.belief.table import BeliefTable, Level, _rebalance
from openhearts.engine import cards
from openhearts.engine.game import deal
from openhearts.engine.kernel import jit_enabled, rebalance, rebalance_iters
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.players.random_player import RandomPlayer

pytestmark = pytest.mark.skipif(not jit_enabled(),
                                reason="JIT disabled in this environment")

COARSE_AFTER = 2000


def _corpus(n_games=40):
    """Raw (pre-rebalance) inputs from every decision point of `n_games`.

    Mixed Random/Heuristic prefixes so the void patterns differ game to game;
    every trick number 0-12 is represented because we snapshot at each play.
    """
    out = []
    for seed in range(n_games):
        rng = np.random.default_rng(seed)
        players = [RandomPlayer(rng) if (seed + i) % 2 else HeuristicPlayer()
                   for i in range(4)]
        state = deal(np.random.default_rng(1000 + seed))
        while not state.is_over():
            seat = state.to_play
            view = state.view_for(seat)
            probs, _voids, hand_sizes, _opps, unseen = T._build_raw(
                view, Level.FULL)
            out.append((probs, hand_sizes, unseen))
            state.play(players[seat].choose(view))
    return out


@pytest.fixture(scope="module")
def corpus():
    c = _corpus()
    assert len(c) >= 2000, f"corpus too small: {len(c)}"
    return c


def test_bitwise_equal_across_corpus(corpus):
    coarse = 0
    for probs, hand_sizes, unseen in corpus:
        py = _rebalance(probs.copy(), hand_sizes, unseen)
        jt = rebalance(probs.copy(), hand_sizes, unseen)
        assert np.array_equal(py, jt), "rebalance output differs bitwise"
        if rebalance_iters(probs.copy(), hand_sizes, unseen) >= COARSE_AFTER:
            coarse += 1
    # if this fires the corpus is wrong, not the kernel: without boundary-zero
    # views the bitwise claim above is untested where it matters most.
    assert coarse > 0, "corpus never reached the coarse-tolerance path"


def test_iteration_count_matches_python(corpus):
    """Bracket the Python loop's pass count with the kernel's, via max_iters."""
    picked = []
    for probs, hand_sizes, unseen in corpus:
        k = rebalance_iters(probs.copy(), hand_sizes, unseen)
        if k >= COARSE_AFTER and not any(x[3] >= COARSE_AFTER for x in picked):
            picked.append((probs, hand_sizes, unseen, k))
        elif 1 < k < 100 and len(picked) < 6:
            picked.append((probs, hand_sizes, unseen, k))
        if len(picked) >= 6 and any(x[3] >= COARSE_AFTER for x in picked):
            break
    assert any(x[3] >= COARSE_AFTER for x in picked)
    for probs, hand_sizes, unseen, k in picked:
        _rebalance(probs.copy(), hand_sizes, unseen, max_iters=k)
        with pytest.raises(AssertionError):
            _rebalance(probs.copy(), hand_sizes, unseen, max_iters=k - 1)


def test_no_unseen_columns_returns_input_untouched():
    probs = np.zeros((3, 52))
    probs[0, 0] = 0.5
    assert _rebalance(probs, [1, 1, 1], 0) is probs
    assert rebalance(probs, [1, 1, 1], 0) is probs


def test_column_collapse_raises_both():
    # card 0's only possible holder is opponent 0, who holds no cards: the
    # row scale drives that entry to zero and the column collapses.
    probs = np.zeros((3, 52))
    probs[0, 0] = 1.0
    probs[:, 1] = 1.0
    unseen = cards.bit(0) | cards.bit(1)
    hand_sizes = [0, 1, 1]
    with pytest.raises(AssertionError) as py_err:
        _rebalance(probs.copy(), hand_sizes, unseen)
    with pytest.raises(AssertionError) as jit_err:
        rebalance(probs.copy(), hand_sizes, unseen)
    assert "column collapsed to zero during rebalance" in str(py_err.value)
    assert str(jit_err.value) == str(py_err.value)


def test_non_convergence_raises_both(corpus):
    probs, hand_sizes, unseen = next(
        c for c in corpus
        if rebalance_iters(c[0].copy(), c[1], c[2]) > 3)
    with pytest.raises(AssertionError) as py_err:
        _rebalance(probs.copy(), hand_sizes, unseen, max_iters=1)
    with pytest.raises(AssertionError) as jit_err:
        rebalance(probs.copy(), hand_sizes, unseen, max_iters=1)
    assert "rebalance did not converge" in str(py_err.value)
    assert str(jit_err.value) == str(py_err.value)


def test_input_array_not_mutated(corpus):
    probs, hand_sizes, unseen = corpus[20]
    before = probs.copy()
    rebalance(probs, hand_sizes, unseen)
    assert np.array_equal(probs, before)


def test_from_view_dispatch_matches_no_jit(tmp_path):
    """End-to-end: BeliefTable.from_view is bitwise identical in both modes."""
    script = tmp_path / "dump.py"
    script.write_text(
        "import numpy as np\n"
        "from openhearts.belief.table import BeliefTable, Level\n"
        "from openhearts.engine.game import deal\n"
        "from openhearts.players.heuristic import HeuristicPlayer\n"
        "out = []\n"
        "for seed in range(6):\n"
        "    st = deal(np.random.default_rng(2000 + seed))\n"
        "    p = HeuristicPlayer()\n"
        "    while not st.is_over():\n"
        "        v = st.view_for(st.to_play)\n"
        "        out.append(BeliefTable.from_view(v, Level.FULL).probs)\n"
        "        st.play(p.choose(v))\n"
        "np.save(__import__('sys').argv[1], np.array(out))\n"
    )
    paths = {}
    for tag, no_jit in (("jit", "0"), ("nojit", "1")):
        env = dict(os.environ, OPENHEARTS_NO_JIT=no_jit)
        paths[tag] = str(tmp_path / f"{tag}.npy")
        subprocess.run([sys.executable, str(script), paths[tag]],
                       env=env, check=True)
    a, b = np.load(paths["jit"]), np.load(paths["nojit"])
    assert np.array_equal(a, b)


def test_from_view_still_satisfies_constraints():
    state = deal(np.random.default_rng(11))
    for _ in range(20):
        state.play(HeuristicPlayer().choose(state.view_for(state.to_play)))
    t = BeliefTable.from_view(state.view_for(state.to_play), Level.FULL)
    np.testing.assert_allclose(t.probs.sum(axis=1), t.hand_sizes, atol=1e-6)
    for c in cards.cards_in(t.unseen_mask):
        np.testing.assert_allclose(t.probs[:, c].sum(), 1.0, atol=1e-6)
