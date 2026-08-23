"""Phase 2.8 gates: the FUSED honest-search decision must be BITWISE identical.

Naming note: `tests/test_jit_fused.py` already belongs to Phase 3.6 (the fused
draw+audit kernel), so this file is `test_jit_fused_decision.py`.

What "bitwise" means here, and why it is achievable. The fused path does not
seed numba's generator once per decision (that was the parked design, and it
changed the rng stream). Instead the orchestrator PRE-DRAWS the inner seeds
with the same scalar `rng.integers(2**63)` call the unfused path uses, the
kernel consumes them in the same order, and the Generator is rewound and
advanced by exactly the number consumed. Both means are an integer sum divided
once, so there is no float summation order to disagree about either. Anything
short of exact equality -- of the card, of every per-candidate mean's float64
bits, of the fallback counters, of the rng state after the decision -- is a
bug, not a tolerance question.

Gate 1 (evidence port), gate 2 here restated as BITWISE (the plan block only
asked for statistical pinning; the pre-drawn-seed redesign lets us demand
more), gates 3 and 4 are the suite-wide default-off / NO_JIT runs and the
timing report.
"""
import numpy as np
import pytest

from openhearts.belief.table import BeliefTable, Level
from openhearts.engine import cards, kernel
from openhearts.engine.game import deal
from openhearts.engine.kernel import jit_enabled
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.search.decision import state_from_view
from openhearts.search.exploiter import (ExploiterSearchPlayer,
                                         champion_model_factory)
from openhearts.search.honest import HonestSearchPlayer

pytestmark = pytest.mark.skipif(not jit_enabled(),
                                reason="JIT disabled in this environment")

# Corpus configs. FULL 10x5 is the champion's shape at test speed; the
# UNIFORM row is the HOSTILE config -- a table that ignores voids feeding a
# sampler that respects them dead-ends often, which is the only cheap way to
# make the inner sampler actually fail and fall back.
CORPUS_FULL = (Level.FULL, 10, 5)
CORPUS_HOSTILE = (Level.UNIFORM, 10, 5)
N_GAMES_FULL = 10
N_GAMES_HOSTILE = 8


def _play_corpus(level, n_outer, n_inner, n_games, seed0):
    """Champion-vs-champion games; keep every view with a real choice."""
    views = []
    for g in range(n_games):
        rng = np.random.default_rng(seed0 + g)
        players = [HonestSearchPlayer(level, n_outer, n_inner, rng)
                   for _ in range(4)]
        state = deal(np.random.default_rng(50000 + g))
        while not state.is_over():
            seat = state.to_play
            view = state.view_for(seat)
            if len(cards.cards_in(view.legal_moves)) > 1:
                views.append((view, (level, n_outer, n_inner)))
            state.play(players[seat].choose(view))
    return views


@pytest.fixture(scope="module")
def corpus():
    out = (_play_corpus(*CORPUS_FULL, N_GAMES_FULL, 1000)
           + _play_corpus(*CORPUS_HOSTILE, N_GAMES_HOSTILE, 2000))
    assert len(out) >= 500, f"corpus too small: {len(out)}"
    tricks = {v.trick_number for v, _cfg in out}
    # A 13-trick hand's last trick is forced, so trick 12 never has a choice.
    assert tricks >= set(range(12)), f"tricks not covered: {sorted(tricks)}"
    return out


def _reference_decision(view, level, n_outer, n_inner, rng):
    """Verbatim mirror of `HonestSearchPlayer.choose` that also returns the
    per-candidate means (which `choose` does not expose).

    Validated by the gate itself: the same test asserts this mirror picks the
    same card as the real unfused `choose` under the same seed, so a drifted
    mirror fails loudly rather than validating the kernel against itself.
    """
    player = HonestSearchPlayer(level, n_outer, n_inner, rng, fused=False)
    legal = cards.cards_in(view.legal_moves)
    table = BeliefTable.from_view(view, level)
    arrangements = player._sample(table, n_outer)
    if len(arrangements) * 2 < n_outer:
        return HeuristicPlayer().choose(view), None, player
    base = view.scores[view.seat]
    avgs = []
    for card in legal:
        total = 0
        for hands in arrangements:
            state = state_from_view(view, hands)
            state.play(card)
            player._playout(state, view.seat)
            total += state.scores[view.seat] - base
        avgs.append(total / len(arrangements))
    best = legal[int(np.argmin(np.array(avgs)))]  # argmin: first minimum wins
    return best, np.array(avgs, dtype=np.float64), player


def _run_one(view, cfg, seed, fused):
    """One decision through the real player; return everything observable."""
    level, n_outer, n_inner = cfg
    rng = np.random.default_rng(seed)
    p = HonestSearchPlayer(level, n_outer, n_inner, rng, fused=fused)
    card = p.choose(view)
    return {
        "card": card,
        "state": rng.bit_generator.state,
        "inner_fallbacks": p.inner_fallbacks,
        "inner_failed_samples": p.inner_failed_samples,
        "fallbacks": p.fallbacks,
        "failed_samples": p.failed_samples,
        "avgs": p.last_avgs,
    }


def test_fused_is_bitwise_identical(corpus):
    """The headline gate: card, means, counters and rng state, all exact."""
    n_inner_fb = n_inner_failed = n_outer_fb = 0
    n_with_means = 0
    for i, (view, cfg) in enumerate(corpus):
        seed = 900000 + i
        unf = _run_one(view, cfg, seed, fused=False)
        fus = _run_one(view, cfg, seed, fused=True)

        assert fus["card"] == unf["card"], f"card differs at decision {i}"
        assert fus["state"] == unf["state"], f"rng state differs at {i}"
        for k in ("inner_fallbacks", "inner_failed_samples",
                  "fallbacks", "failed_samples"):
            assert fus[k] == unf[k], f"{k} differs at decision {i}"

        n_inner_fb += unf["inner_fallbacks"]
        n_inner_failed += unf["inner_failed_samples"]
        n_outer_fb += unf["fallbacks"]

        if unf["fallbacks"]:
            # Outer fallback: both paths return the heuristic's card BEFORE
            # the fused branch is reached, so there are no means to compare.
            assert fus["avgs"] is None
            continue

        ref_card, ref_avgs, _p = _reference_decision(
            view, *cfg, np.random.default_rng(seed))
        assert ref_card == unf["card"], (
            f"the test's mirror of choose() drifted at decision {i}")
        assert fus["avgs"] is not None
        assert np.array_equal(fus["avgs"], ref_avgs), (
            f"per-candidate means differ at decision {i}: "
            f"{fus['avgs']} vs {ref_avgs}")
        # Exact float equality, spelled out: array_equal on float64 is bitwise
        # except for NaN, and a NaN mean would itself be a bug.
        assert not np.isnan(fus["avgs"]).any()
        n_with_means += 1

    assert n_with_means >= 500, f"only {n_with_means} decisions compared means"
    print(f"\n[gate 2] {len(corpus)} decisions, {n_with_means} with means; "
          f"inner fallbacks {n_inner_fb}, failed inner samples "
          f"{n_inner_failed}, outer fallbacks {n_outer_fb}")


def test_fused_is_bitwise_identical_when_the_sampler_fails(corpus,
                                                           monkeypatch):
    """The same gate, on the branches ordinary play almost never reaches.

    Sampler failures are genuinely rare at `max_restarts=200`: measured over
    the corpus above, roughly 1 draw in 40,000 fails, and the inner FALLBACK
    (`2 * n_ok < n_inner`, i.e. most of a decision's inner draws failing) then
    essentially never fires. Rather than leave that branch unproven, we lower
    the restart cap to 2 -- for BOTH paths identically, so bitwise is still
    the standard -- which makes failures common and the fallback frequent.
    Nothing in the shipped code changes; only the two kernel entry points'
    `max_restarts` argument is bound differently for this test.
    """
    orig_sample = kernel.sample_arrangements
    orig_decision = kernel.honest_decision
    monkeypatch.setattr(
        kernel, "sample_arrangements",
        lambda t, r, n, max_restarts=2: orig_sample(t, r, n, 2))
    monkeypatch.setattr(
        kernel, "honest_decision",
        lambda *a, max_restarts=2, **k: orig_decision(*a, max_restarts=2, **k))

    n_inner_fb = n_inner_failed = n_outer_fb = n_cmp = 0
    for i, (view, cfg) in enumerate(corpus):
        if cfg[0] != Level.UNIFORM:
            continue  # the hostile half of the corpus only, for speed
        seed = 700000 + i
        unf = _run_one(view, cfg, seed, fused=False)
        fus = _run_one(view, cfg, seed, fused=True)
        assert fus["card"] == unf["card"], f"card differs at {i}"
        assert fus["state"] == unf["state"], f"rng state differs at {i}"
        for k in ("inner_fallbacks", "inner_failed_samples",
                  "fallbacks", "failed_samples"):
            assert fus[k] == unf[k], f"{k} differs at decision {i}"
        n_inner_fb += unf["inner_fallbacks"]
        n_inner_failed += unf["inner_failed_samples"]
        n_outer_fb += unf["fallbacks"]
        if not unf["fallbacks"]:
            ref_card, ref_avgs, _p = _reference_decision(
                view, *cfg, np.random.default_rng(seed))
            assert ref_card == unf["card"]
            assert np.array_equal(fus["avgs"], ref_avgs), (
                f"per-candidate means differ at decision {i}")
            n_cmp += 1

    print(f"\n[gate 2, hostile] {n_cmp} decisions with means; inner fallbacks "
          f"{n_inner_fb}, failed inner samples {n_inner_failed}, outer "
          f"fallbacks {n_outer_fb}")
    assert n_inner_fb >= 1, "never hit an inner sampler fallback"
    assert n_inner_failed >= 1, "never hit a failed inner sample"
    assert n_outer_fb >= 1, "never hit an outer sampler fallback"
    assert n_cmp >= 100


# --------------------------------------------------------------------------
# gate 1: the in-kernel evidence extraction
# --------------------------------------------------------------------------
_LEVEL_CODE = {Level.UNIFORM: kernel.LEVEL_UNIFORM,
               Level.VOIDS: kernel.LEVEL_VOIDS,
               Level.FULL: kernel.LEVEL_FULL}


def _imagined_states(corpus, limit=2600):
    """Views taken at the INTERCEPTION POINT inside imagined playouts.

    These -- not real-game views -- are what the fused kernel's `_evidence`
    actually sees: a real prefix, our candidate, and a stretch of invented
    plays on top. Harvested through the same kernel entry the player uses.
    """
    out = []
    rng = np.random.default_rng(4242)
    for view, cfg in corpus:
        level, n_outer, _n_inner = cfg
        table = BeliefTable.from_view(view, level)
        arrangements, _nf = kernel.sample_arrangements(table, rng, 2)
        for hands in arrangements:
            for card in cards.cards_in(view.legal_moves)[:2]:
                state = state_from_view(view, hands)
                state.play(card)
                if kernel.run_playout_until_decision(state, view.seat):
                    out.append((state.view_for(view.seat), level))
        if len(out) >= limit:
            break
    return out


@pytest.fixture(scope="module")
def imagined(corpus):
    states = _imagined_states(corpus)
    assert len(states) >= 2000, f"too few imagined states: {len(states)}"
    return states


def test_evidence_port_bitwise(imagined):
    coarse = 0
    levels = set()
    for i, (view, level) in enumerate(imagined):
        # The load-bearing invariant of the flat play sequence: history only
        # ever grows by whole tricks, so runs of 4 ARE the tricks.
        assert len(view.history) % 4 == 0

        ref = BeliefTable.from_view(view, level)
        probs, voids, hand_sizes, unseen_mask = kernel.honest_evidence(
            view.hand, list(view.history) + list(view.current_trick),
            view.seat, _LEVEL_CODE[level], True)

        assert np.array_equal(probs, ref.probs), f"probs differ at {i}"
        assert probs.dtype == ref.probs.dtype
        assert voids == [set(s) for s in ref.voids], f"voids differ at {i}"
        assert hand_sizes == list(ref.hand_sizes), f"hand sizes differ at {i}"
        assert unseen_mask == ref.unseen_mask, f"unseen differs at {i}"

        levels.add(level)
        if level == Level.FULL:
            # Same coarse-path counter `test_jit_rebalance.py` uses: an IPF
            # that needs >= coarse_after passes is sitting on the simplex
            # boundary, i.e. a combinatorially FORCED zero.
            if kernel.rebalance_iters(*_raw_for(view, Level.FULL)) >= 2000:
                coarse += 1
    assert len(levels) >= 2, "only one belief level in the evidence corpus"
    print(f"\n[gate 1] {len(imagined)} imagined states bitwise-equal; "
          f"{coarse} needed the coarse-tolerance (boundary-zero) path")
    assert coarse >= 1, (
        "no boundary-zero rebalance in the corpus: the hard branch of the "
        "port is unproven")


def _raw_for(view, level):
    from openhearts.belief import table as T
    probs, _v, hand_sizes, _o, unseen = T._build_raw(view, level)
    return probs, hand_sizes, unseen


def test_evidence_respects_voids_flag(imagined):
    """`sampler_respects_voids=False` empties the void sets ONLY.

    That is exactly what `HonestSearchPlayer.choose` does to the table: it
    rebuilds it with empty void sets and the SAME probs, so the zeroing a
    VOIDS/FULL table already baked in stays baked in.
    """
    checked = 0
    for view, level in imagined:
        p_on, v_on, _h, _u = kernel.honest_evidence(
            view.hand, list(view.history) + list(view.current_trick),
            view.seat, _LEVEL_CODE[level], True)
        p_off, v_off, _h2, _u2 = kernel.honest_evidence(
            view.hand, list(view.history) + list(view.current_trick),
            view.seat, _LEVEL_CODE[level], False)
        assert np.array_equal(p_on, p_off)
        assert v_off == [set(), set(), set()]
        if any(v_on):
            checked += 1
    assert checked >= 1, "no voids anywhere in the corpus"


def test_fused_is_bitwise_identical_with_a_posterior():
    """The choice-aware path fuses too, and just as exactly.

    The posterior supplies the OUTER worlds and does its own rng draws BEFORE
    the fused branch is reached, so the kernel sees the same arrangements from
    either source. Reading is retired as a bot feature (Phase 6A), but the
    socket is live code and the gate should cover it.
    """
    from openhearts.belief.weighted import WeightedPosterior

    def factory(view, rng):
        return WeightedPosterior.from_view(
            view, Level.FULL, HeuristicPlayer(), 0.0, 8, rng=rng,
            max_draws=20000, keep_worlds=True)

    n = 0
    state = deal(np.random.default_rng(31))
    heur = HeuristicPlayer()
    while not state.is_over():
        seat = state.to_play
        view = state.view_for(seat)
        card = heur.choose(view)
        if seat == 0 and len(cards.cards_in(view.legal_moves)) > 1:
            out = []
            for fused in (False, True):
                rng = np.random.default_rng(555 + n)
                p = HonestSearchPlayer(Level.FULL, 8, 3, rng,
                                       posterior_factory=factory, fused=fused)
                out.append((p.choose(view), rng.bit_generator.state,
                            p.inner_fallbacks, p.inner_failed_samples,
                            p.posterior_worlds, p.posterior_collapses))
            assert out[0] == out[1], f"posterior-path decision {n} differs"
            # Without this the gate is decorative: if the posterior collapsed
            # every time, `arrangements` comes back None, the plain constraint
            # sampler runs, and the two sides agree while proving nothing
            # about the posterior -> fused handoff.
            assert out[0][4] > 0, (
                f"decision {n} never got worlds from the posterior")
            card = out[0][0]
            n += 1
        state.play(card)
    assert n >= 8, f"only {n} posterior decisions exercised"


# --------------------------------------------------------------------------
# the flag itself
# --------------------------------------------------------------------------
def test_default_is_off():
    p = HonestSearchPlayer(Level.FULL, 4, 2, np.random.default_rng(0))
    assert p.fused is False
    assert p._use_fused() is False


def test_fused_off_for_python_sampler():
    """`jit_sampler=False` must NOT fuse: the unfused inner would use the
    Python sampler's per-card rng stream, which the kernel does not reproduce.
    """
    p = HonestSearchPlayer(Level.FULL, 4, 2, np.random.default_rng(0),
                           jit_sampler=False, fused=True)
    assert p._use_fused() is False


def test_fused_off_without_inner():
    p = HonestSearchPlayer(Level.FULL, 4, 0, np.random.default_rng(0),
                           fused=True)
    assert p._use_fused() is False


def test_exploiter_refuses_fused():
    """The exploiter's whole mechanism is a Python `_playout` hook that the
    compiled loop cannot see, so a fused exploiter would silently measure
    plain honest-FULL. Two locks: the constructor and `_use_fused`.
    """
    with pytest.raises(ValueError, match="fused"):
        ExploiterSearchPlayer(Level.FULL, 4, 2, np.random.default_rng(0),
                              fused=True)
    ex = ExploiterSearchPlayer(Level.FULL, 4, 2, np.random.default_rng(0))
    ex.fused = True                      # every other route in
    assert ex._use_fused() is False

    factory = champion_model_factory(Level.FULL, 3, 2, fused=True)
    nested = factory(0, np.random.default_rng(1))
    assert nested._use_fused() is True   # the nested champion MAY fuse
