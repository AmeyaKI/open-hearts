"""Task 5: value-truncated honest search.

The load-bearing test is the FIRST one: `horizon=None` must reduce EXACTLY to
today's `HonestSearchPlayer` -- same rng stream, same chosen card, on >=200
sampled decisions spread across the whole hand. Same discipline as Phase 2's
`n_inner=0` reduction test: a refactor that quietly changes the search is a
silent invalidation of every published row, so it gets pinned before anything
new is measured.

The second pin is the TERMINAL one: when the imagined hand ends before the
horizon, the value net must never be consulted and the score must be the
actual points -- so `horizon=k` on a position with <=k tricks left is the full
playout, bitwise. That is checked two ways: chosen card equal to
`HonestSearchPlayer`, and a `value_calls` counter that must be exactly 0.
"""
import os

import numpy as np
import pytest

from openhearts.belief.table import Level
from openhearts.engine import cards, kernel
from openhearts.engine.game import deal, play_game
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.search.honest import HonestSearchPlayer
from openhearts.search.valuesearch import ValueSearchPlayer

WEIGHTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "value_v1.npz")

FAST = not kernel.jit_enabled()


def _decisions(seeds, min_trick=0):
    """Replay heuristic games; yield (view, ply_index) at real decisions."""
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        state = deal(rng)
        h = HeuristicPlayer()
        while not state.is_over():
            seat = state.to_play
            view = state.view_for(seat)
            if (len(cards.cards_in(view.legal_moves)) > 1
                    and view.trick_number >= min_trick):
                out.append(view)
            state.play(h.choose(view))
    return out


# --------------------------------------------------------------- (a) horizon=None
def test_horizon_none_reduces_to_honest_search():
    views = _decisions(range(40, 60))
    # spread across the whole hand: stride-sample rather than take a prefix
    step = max(1, len(views) // 220)
    views = views[::step][:220]
    assert len(views) >= 200, f"only {len(views)} decisions sampled"

    n_outer, n_inner = (3, 1) if FAST else (5, 2)
    for i, view in enumerate(views):
        a = ValueSearchPlayer(Level.FULL, n_outer=n_outer, n_inner=n_inner,
                              rng=np.random.default_rng(1000 + i),
                              horizon=None, weights_path=WEIGHTS).choose(view)
        b = HonestSearchPlayer(Level.FULL, n_outer=n_outer, n_inner=n_inner,
                               rng=np.random.default_rng(1000 + i)).choose(view)
        assert a == b, f"decision {i}: horizon=None chose {a}, honest chose {b}"


# ------------------------------------------------------------- (b) legal games
@pytest.mark.parametrize("horizon", [0, 1, 2])
@pytest.mark.parametrize("seed", [3, 17, 42])
def test_games_complete_legally(horizon, seed):
    rng = np.random.default_rng(seed)
    bot = ValueSearchPlayer(Level.FULL, n_outer=3, n_inner=1, rng=rng,
                            horizon=horizon, weights_path=WEIGHTS)
    state = play_game(deal(rng), [bot] + [HeuristicPlayer() for _ in range(3)])
    assert sum(state.scores) == 26


# ------------------------------------------------------------ (c) JIT vs NO_JIT
@pytest.mark.skipif(not kernel.jit_enabled(), reason="needs the JIT available")
@pytest.mark.parametrize("n_inner", [0, 1])
@pytest.mark.parametrize("horizon", [0, 1])
def test_jit_and_python_paths_agree(monkeypatch, horizon, n_inner):
    # n_inner=0 exercises the FUSED batch kernel at the outer stage;
    # n_inner=1 exercises the per-world outer path with a fused inner.
    # horizon=0 is the ONLY setting that stops mid-trick (trick_number moves
    # only at trick completion), so it is the only test of the featurizer's
    # empty-vs-nonempty trick handling on both paths -- the Python reference
    # guards `_trick_head` explicitly and the kernel does not.
    views = _decisions(range(70, 78))
    step = max(1, len(views) // 55)
    views = views[::step][:55]
    assert len(views) >= 50, f"only {len(views)} decisions sampled"

    def run():
        # jit_sampler=False so BOTH modes draw the identical (Python-sampler)
        # arrangements; the only thing under test is the scoring path.
        return [ValueSearchPlayer(Level.FULL, n_outer=3, n_inner=n_inner,
                                  rng=np.random.default_rng(2000 + i),
                                  horizon=horizon, weights_path=WEIGHTS,
                                  jit_sampler=False).choose(v)
                for i, v in enumerate(views)]

    jit_choices = run()
    monkeypatch.setenv("OPENHEARTS_NO_JIT", "1")
    kernel.reset_jit_enabled()
    try:
        py_choices = run()
    finally:
        monkeypatch.delenv("OPENHEARTS_NO_JIT", raising=False)
        kernel.reset_jit_enabled()
    assert py_choices == jit_choices


# ------------------------------------------------------- (d) terminal accounting
@pytest.mark.parametrize("horizon", [2, 3])
def test_horizon_past_end_is_exactly_full_playout(horizon):
    """Positions with <= horizon tricks left: the net is never consulted and
    the score is the real points, so the choice must match full playouts."""
    views = [v for v in _decisions(range(80, 100))
             if v.trick_number >= 13 - horizon]
    assert len(views) >= 20, f"only {len(views)} late decisions"
    views = views[:30]
    for i, view in enumerate(views):
        p = ValueSearchPlayer(Level.FULL, n_outer=3, n_inner=1,
                              rng=np.random.default_rng(3000 + i),
                              horizon=horizon, weights_path=WEIGHTS)
        a = p.choose(view)
        b = HonestSearchPlayer(Level.FULL, n_outer=3, n_inner=1,
                               rng=np.random.default_rng(3000 + i)).choose(view)
        assert p.value_calls == 0, (
            f"net consulted {p.value_calls} times on a position with "
            f"{13 - view.trick_number} tricks left and horizon={horizon}")
        assert a == b, f"late decision {i}: {a} != {b}"


def test_value_calls_counted_when_horizon_bites():
    """Sanity companion to the test above: mid-hand, the net IS consulted."""
    view = _decisions([81])[0]
    p = ValueSearchPlayer(Level.FULL, n_outer=4, n_inner=1,
                          rng=np.random.default_rng(5), horizon=1,
                          weights_path=WEIGHTS)
    p.choose(view)
    assert p.value_calls > 0


# ------------------------------------------ (e) the horizon kernel vs the old one
@pytest.mark.skipif(not kernel.jit_enabled(), reason="needs the JIT")
def test_run_horizon_kernel_matches_playout_to_end():
    """`_run_horizon` with no horizon must be `playout_to_end`, exactly: it
    reuses kernel._legal/_choose/_trick_head, so only the loop scaffolding is
    new, and this pins that."""
    from openhearts.search import valuesearch as vs
    for seed in range(20):
        rng = np.random.default_rng(seed)
        state = deal(rng)
        args = kernel._to_arrays(state)
        hands, scores, tc, ts, tl, played = args
        oc, os_ = np.zeros(52, np.int64), np.zeros(52, np.int64)
        sc1 = scores.copy()
        r1 = kernel.playout_to_end(hands.copy(), state.to_play, played,
                                   tc.copy(), ts.copy(), tl,
                                   state.hearts_broken, state.trick_number,
                                   sc1, oc, os_)
        oc2, os2 = np.zeros(52, np.int64), np.zeros(52, np.int64)
        sc2 = scores.copy()
        r2 = vs._run_horizon(hands.copy(), state.to_play, played, tc.copy(),
                             ts.copy(), tl, state.hearts_broken,
                             state.trick_number, sc2, -1, -1, oc2, os2)
        assert r1[0] == r2[0] == 0
        assert r1[6] == r2[6]
        assert list(oc[:r1[6]]) == list(oc2[:r2[6]])
        assert list(os_[:r1[6]]) == list(os2[:r2[6]])
        assert list(sc1) == list(sc2)


# ------------------------------------------- (f) which output slot is our seat
def test_net_output_slot_0_is_the_evaluated_seat():
    """The whole search hangs on `v[0]` being the EVALUATED seat's remaining
    points. If that index were wrong -- slot 1, or the rotation inverted --
    every other test here still passes (games complete, JIT==NO_JIT because
    both paths share the bug, and the terminal test never calls the net); the
    only symptom would be mediocre agreement, which is exactly what the Task-3
    MSE null predicts anyway. So the indexing gets its own pin.

    Method: at real mid-hand positions, take the net's four outputs and the
    points each seat ACTUALLY goes on to take under a full heuristic playout
    of that same position. Slot r must match absolute seat (seat + r) % 4
    better than it matches any other seat.
    """
    from openhearts.engine.features import featurize
    from openhearts.value.infer import load_weights, value_forward

    weights = load_weights(WEIGHTS)
    preds, actuals = [], []
    for seed in range(200, 240):
        rng = np.random.default_rng(seed)
        state = deal(rng)
        h = HeuristicPlayer()
        stop = 4 + (seed % 40)          # a spread of mid-hand positions
        for _ in range(stop):
            state.play(h.choose(state.view_for(state.to_play)))
        seat = state.to_play
        hands, scores, tc, ts, tl, played = kernel._to_arrays(state)
        led, _wr, win = kernel._trick_head(tc, ts, tl)
        f = featurize(hands, played, tc, ts, tl, led, win,
                      state.hearts_broken, state.trick_number, scores, seat)
        v = value_forward(*weights, f)
        before = list(state.scores)
        finished = state.copy()
        while not finished.is_over():
            finished.play(h.choose(finished.view_for(finished.to_play)))
        rest = [finished.scores[a] - before[a] for a in range(4)]
        preds.append([v[r] for r in range(4)])
        # rotated to the evaluated seat, same convention as the featurizer
        actuals.append([rest[(seat + r) % 4] for r in range(4)])

    preds = np.asarray(preds)
    actuals = np.asarray(actuals)
    assert len(preds) >= 40
    mse = np.zeros((4, 4))
    for r in range(4):
        for j in range(4):
            mse[r, j] = float(np.mean((preds[:, r] - actuals[:, j]) ** 2))
    for r in range(4):
        assert int(np.argmin(mse[r])) == r, (
            f"net slot {r} matches rotated seat {int(np.argmin(mse[r]))} "
            f"better than seat {r}; the rotation/index convention is wrong.\n"
            f"MSE matrix:\n{mse}")
    # Sanity on the magnitude: Task 3 reported test MSE 18.51 for the
    # evaluated seat against a constant-predictor baseline of 29.12. This is a
    # different (on-policy, mid-hand) sample, so the bound is loose on
    # purpose -- it only has to exclude "the net is not predicting points".
    assert mse[0, 0] < 29.12, f"slot 0 MSE {mse[0, 0]:.2f} is no better than "\
        "the constant-predictor baseline"
