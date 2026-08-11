"""Phase 3.6: the fused draw+audit kernel must be bitwise-identical to 3.5.

The 3.5 JIT path drew a chunk of candidates with `kernel.sample_arrangements`
and then audited them one at a time through Python glue
(`_reconstruct_original_hands` + `kernel.audit_world`). 3.6 moves the whole
chunk -- draw, reconstruct, audit, accumulate -- into one compiled call.

Equality target: BITWISE. The fused path keeps the 3.5 chunk loop
(`chunk = min(max_draws - draws, n_worlds - kept)`, one `rng.integers(2**63)`
seed draw per chunk) and calls the SAME `sample_arrangements_batch` inside the
kernel, so the numba RNG stream, the candidate worlds, the weights and the
accumulation order are all unchanged.
"""
import numpy as np
import pytest

from openhearts.belief.table import BeliefTable, Level
from openhearts.belief.weighted import WeightedPosterior, world_weight
from openhearts.engine import cards, kernel
from openhearts.engine.game import deal
from openhearts.players.heuristic import HeuristicPlayer

pytestmark = pytest.mark.skipif(not kernel.jit_enabled(),
                                reason="fused path requires the JIT")


def _mid_game_view(seed, n_plies):
    rng = np.random.default_rng(seed)
    state = deal(rng)
    players = [HeuristicPlayer() for _ in range(4)]
    observer = state.to_play
    for _ in range(n_plies):
        s = state.to_play
        state.play(players[s].choose(state.view_for(s)))
    return state, state.view_for(observer), observer


@pytest.mark.parametrize("seed,plies", [(11, 20), (12, 32), (13, 28),
                                        (14, 40)])
def test_fused_posterior_bitwise_matches_unfused(seed, plies):
    _state, view, _obs = _mid_game_view(seed, plies)
    kw = dict(level=Level.FULL, policy=HeuristicPlayer(), epsilon=0.0,
              n_worlds=50, max_draws=20000)
    a = WeightedPosterior.from_view(view, rng=np.random.default_rng(7),
                                    _fused=True, **kw)
    b = WeightedPosterior.from_view(view, rng=np.random.default_rng(7),
                                    _fused=False, **kw)
    assert np.array_equal(a.probs, b.probs)
    assert a.total_weight == b.total_weight
    assert a.n_effective == b.n_effective
    assert a.draws_used == b.draws_used
    assert a.n_worlds_used == b.n_worlds_used


@pytest.mark.parametrize("seed,plies,eps", [(21, 24, 0.0), (22, 36, 0.1),
                                            (23, 16, 0.3)])
def test_fused_kept_worlds_and_weights_identical(seed, plies, eps):
    """Per-world comparison: same kept worlds, same weights, same order."""
    _state, view, _obs = _mid_game_view(seed, plies)
    policy = HeuristicPlayer()
    table = BeliefTable.from_view(view, Level.FULL)
    n_worlds, max_draws = 40, 5000

    def unfused_records():
        rng = np.random.default_rng(99)
        recs, draws, kept = [], 0, 0
        while kept < n_worlds and draws < max_draws:
            chunk = min(max_draws - draws, n_worlds - kept)
            batch, _nf = kernel.sample_arrangements(table, rng, chunk)
            draws += chunk
            for wh in batch:
                w = world_weight(view, wh, policy, eps)
                if w > 0.0:
                    recs.append(([int(x) for x in wh], w))
                    kept += 1
        return recs, draws

    def fused_records():
        rng = np.random.default_rng(99)
        ctx = kernel.fused_audit_context(view, table, view.seat, eps)
        recs, draws, kept = [], 0, 0
        out_probs = np.zeros((3, 52))
        while kept < n_worlds and draws < max_draws:
            chunk = min(max_draws - draws, n_worlds - kept)
            worlds, weights, n_kept, _tw, _tw2 = kernel.draw_audit_chunk(
                ctx, rng, chunk, out_probs)
            draws += chunk
            for j in range(n_kept):
                recs.append(([int(x) for x in worlds[j]], float(weights[j])))
            kept += n_kept
        return recs, draws

    ur, ud = unfused_records()
    fr, fd = fused_records()
    assert ud == fd
    assert len(ur) == len(fr) and len(ur) > 0
    for (uw, uwt), (fw, fwt) in zip(ur, fr):
        assert uw == fw
        assert uwt == fwt  # bitwise


def test_fused_handles_empty_play_history():
    """Opening lead: no evidence yet, so every drawn world weighs exactly 1.0.

    The only branch of the fused kernel the mid-game cases never reach --
    zero-length `plays_cards` / `plays_seats` must type in numba and take the
    `n_plays == 0` shortcut instead of indexing `plays_seats[0]`.
    """
    state, view, observer = _mid_game_view(51, 0)
    assert not view.history and not view.current_trick
    table = BeliefTable.from_view(view, Level.FULL)
    ctx = kernel.fused_audit_context(view, table, observer, 0.0)
    out_probs = np.zeros((3, 52))
    worlds, weights, n_kept, total_w, _tw2 = kernel.draw_audit_chunk(
        ctx, np.random.default_rng(5), 6, out_probs)
    assert n_kept == 6
    assert list(weights[:6]) == [1.0] * 6
    assert total_w == 6.0
    for j in range(6):
        assert sum(bin(int(x)).count("1") for x in worlds[j]) == 39

    # and the same through the public entry point
    post = WeightedPosterior.from_view(
        view, Level.FULL, HeuristicPlayer(), epsilon=0.0, n_worlds=6,
        rng=np.random.default_rng(5), max_draws=100)
    assert post.n_worlds_used == 6
    assert post.n_effective == 6.0


@pytest.mark.parametrize("seed,plies", [(31, 24), (32, 36), (33, 44)])
def test_truth_world_kept_with_weight_one_at_eps_zero(seed, plies):
    """A truth-pinned proposal draws the true world; fused must keep it at 1.0.

    Isolates the in-kernel reconstruction: probs are the 0/1 truth assignment,
    so the sampler's ascending greedy walk has exactly one completion.
    """
    state, view, observer = _mid_game_view(seed, plies)
    table = BeliefTable.from_view(view, Level.FULL)
    truth = [state.hands[(observer + 1 + i) % 4] for i in range(3)]
    table.probs = np.zeros((3, 52))
    for i in range(3):
        for c in cards.cards_in(truth[i]):
            table.probs[i, c] = 1.0

    ctx = kernel.fused_audit_context(view, table, observer, 0.0)
    out_probs = np.zeros((3, 52))
    worlds, weights, n_kept, total_w, _tw2 = kernel.draw_audit_chunk(
        ctx, np.random.default_rng(5), 3, out_probs)
    assert n_kept == 3
    for j in range(3):
        assert [int(x) for x in worlds[j]] == [int(x) for x in truth]
        assert weights[j] == 1.0
    assert total_w == 3.0
