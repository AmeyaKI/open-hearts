"""Phase 5 Task 5 gates for `opponent/adapt.py`.

TWO CLASSES OF TEST, deliberately separated (the precedent is
`test_profiled_search.py`):

* STRUCTURAL tests run on a SYNTHESIZED conditioned net (seeded random weights
  exported through `npz_io.save_npz`), so they are always green in CI and
  never depend on the lead's training artifacts: pool hygiene (never a
  held-out id), log-space stability over a long match, the weight floor,
  truth-safety of the adapted likelihood, and drop-in compatibility with
  `search/profiled.py`'s factory.

* MEASUREMENT gates (a) identification and (b) held-out benefit need the REAL
  trained CONDITIONED net -- random weights carry no information about who is
  playing, so those gates would be vacuous on synthesized weights. They SKIP
  when `results/profiler_train/profiler_conditioned.npz` is absent (gitignored,
  regenerable by `experiments/train_profiler.py`).

PRE-REGISTERED PROTOCOL for gate (b), written before the run: weights are
fitted on hands 1..h and scored on hand h+1's events -- ALWAYS out of sample.
Scoring the hand that trained the weights would make the gate pass for free.
"Better by hand 3" therefore means: fitted on hands 1-3, scored on hand 4,
mean log-likelihood >= GENERIC's on the same events.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from openhearts.engine.features import NF
from openhearts.opponent.adapt import (BLEND_B, DEFAULT_K, EPS_W,
                                       AdaptedLikelihood, PoolProfiler,
                                       SeatMixture, events_from_history,
                                       pool_ids, train_heldout_ids)
from openhearts.opponent.infer import load_profiler
from openhearts.opponent.npz_io import N_CARDS, save_npz
from openhearts.opponent.params import PARAM_DIM
from openhearts.search.profiled import (ProfilerLikelihood,
                                        profiler_posterior_factory,
                                        profiler_world_logweight)

COND_NPZ = os.path.join(os.path.dirname(__file__), "..", "results",
                        "profiler_train", "profiler_conditioned.npz")
GEN_NPZ = os.path.join(os.path.dirname(__file__), "..", "models",
                       "profiler_v1.npz")
H1, H2 = 24, 16


def _synth(tmp_path, name, n_param_in, seed):
    rng = np.random.default_rng(seed)
    n_in = NF + n_param_in
    w = {"W1": rng.normal(0, .1, (H1, n_in)), "b1": rng.normal(0, .1, H1),
         "W2": rng.normal(0, .1, (H2, H1)), "b2": rng.normal(0, .1, H2),
         "W3": rng.normal(0, .1, (N_CARDS, H2)),
         "b3": rng.normal(0, .1, N_CARDS)}
    path = str(tmp_path / name)
    save_npz(path, w, [n_in, H1, H2, N_CARDS], n_param_in)
    return path


@pytest.fixture(scope="module")
def synth_pool(tmp_path_factory):
    d = tmp_path_factory.mktemp("p5t5")
    cw, _ = load_profiler(_synth(d, "cond.npz", PARAM_DIM, 11))
    gw, _ = load_profiler(_synth(d, "gen.npz", 0, 12))
    return PoolProfiler(cw, pool_ids(8)), ProfilerLikelihood(gw)


def _hand_history(seed, pool):
    """A completed hand's [(seat, card)] plus its table, via Task 2's code."""
    import gen_population_data as G
    from openhearts.engine.game import deal
    table = G._table_for_seed(seed, pool)
    players = [G._make_player(pid, seed, s) for s, pid in enumerate(table)]
    state = deal(np.random.default_rng(seed))
    hist = []
    while not state.is_over():
        s = state.to_play
        c = players[s].choose(state.view_for(s))
        hist.append((s, c))
        state.play(c)
    return hist, table


# --------------------------------------------------------------- structural
def test_pool_is_train_only_and_deterministic():
    train, held = train_heldout_ids()
    ids = pool_ids()
    assert len(ids) == DEFAULT_K
    assert set(ids) <= set(train)
    assert set(ids).isdisjoint(held)
    assert ids == pool_ids(), "pool draw is not deterministic"


def test_events_from_history_matches_the_task2_extractor():
    """`events_from_history` must reproduce the generator's rows exactly."""
    import gen_population_data as G
    seed = 700_123
    hist, _table = _hand_history(seed, None)
    rows, _t = G.play_and_record(seed)
    seats, feats, masks, chosen = events_from_history(hist)
    assert list(seats) == list(rows["acting_seat"])
    assert list(masks) == list(rows["legal_mask"])
    assert list(chosen) == list(rows["chosen_card"])
    assert np.allclose(feats.astype(np.float16),
                       rows["profiler_features"])


def test_weights_stay_finite_and_floored_over_a_long_match(synth_pool):
    """Log space, not products: 12 hands would underflow a naive multiply."""
    pool, _gen = synth_pool
    mix = SeatMixture(pool)
    for i in range(12):
        hist, table = _hand_history(700_500 + i, None)
        mix.observe_hand(hist, table.index(table[0]))
        w = mix.weights
        assert np.isfinite(w).all() and abs(w.sum() - 1) < 1e-9
        assert w.min() > 0.0
    floor = EPS_W / (1 + pool.k * EPS_W)
    assert mix.weights.min() >= floor * (1 - 1e-12)
    assert mix.n_events > 0


def test_reset_restores_the_prior(synth_pool):
    pool, _gen = synth_pool
    mix = SeatMixture(pool)
    hist, _t = _hand_history(700_601, None)
    mix.observe_hand(hist, 0)
    assert not np.allclose(mix.weights, 1.0 / pool.k, atol=1e-9)
    mix.reset()
    assert np.allclose(mix.weights, 1.0 / pool.k, atol=1e-9)
    assert mix.n_events == 0


def test_mixture_probs_are_a_convex_combination(synth_pool):
    pool, _gen = synth_pool
    hist, _t = _hand_history(700_602, None)
    seats, feats, masks, _c = events_from_history(hist)
    pm = pool.member_probs(feats[:5], masks[:5])
    assert pm.shape == (5, pool.k, 52)
    for i in range(5):
        for k in range(pool.k):
            assert abs(pm[i, k].sum() - 1.0) < 1e-9
            # off-mask entries are exactly zero (masked softmax contract)
            for c in range(52):
                if not (int(masks[i]) >> c) & 1:
                    assert pm[i, k, c] == 0.0


@pytest.mark.parametrize("blend", [0.0, BLEND_B])
def test_truth_safety_adapted_probability_is_never_zero(synth_pool, blend):
    """GATE (c): no legal card ever gets exactly 0 under the adapted model."""
    pool, gen = synth_pool
    mixes = {s: SeatMixture(pool) for s in range(4)}
    hist, _t = _hand_history(700_603, None)
    for s in range(4):
        mixes[s].observe_hand(hist, s)
    lik = AdaptedLikelihood(pool, mixes, generic=gen, blend=blend)
    seats, feats, masks, _c = events_from_history(hist)
    P = lik.batch_probs(list(feats), list(seats), list(masks))
    for i in range(P.shape[0]):
        legal = [c for c in range(52) if (int(masks[i]) >> c) & 1]
        assert len(legal) > 1
        assert min(P[i, c] for c in legal) > 0.0
        assert abs(P[i].sum() - 1.0) < 1e-9


def test_adapted_likelihood_is_drop_in_for_the_profiled_factory(synth_pool):
    """Task 6 must be able to build `profiled-RIA` without touching
    `search/profiled.py`'s semantics."""
    from openhearts.engine.game import deal
    pool, gen = synth_pool
    lik = AdaptedLikelihood(pool, {s: SeatMixture(pool) for s in range(4)},
                            generic=gen, blend=BLEND_B)
    assert isinstance(lik, ProfilerLikelihood)
    state = deal(np.random.default_rng(4242))
    for _ in range(9):
        state.play(next(c for c in range(52)
                        if (int(_legal(state)) >> c) & 1))
    view = state.view_for(state.to_play)
    make = profiler_posterior_factory(lik, level="full", n_worlds=8,
                                      max_draws=4000)
    post = make(view, np.random.default_rng(7))
    assert post.n_worlds_used > 0
    assert np.isfinite(post.probs).all()
    # and the world-audit path itself accepts it
    lw = profiler_world_logweight(view, [state.hands[s] for s in
                                         post.opponent_seats], lik)
    assert np.isfinite(lw)


def _legal(state):
    from openhearts.engine.game import legal_moves
    return legal_moves(state.hands[state.to_play], tuple(state.current_trick),
                       state.hearts_broken, state.trick_number)


# -------------------------------------------------------------- measurement
from openhearts.engine.kernel import jit_enabled  # noqa: E402

# The measurement gates push ~1300 network rows through PoolProfiler per
# observed hand. Under OPENHEARTS_NO_JIT=1 that is the pure-Python forward
# loop and the file blows well past its ~60s budget. They are skipped there --
# NOT weakened: no threshold changes, and what they measure (whether the
# trained net can tell personalities apart) has nothing to do with which of
# the two IDENTICAL forward-pass sources ran, an equivalence already pinned by
# tests/test_profiler_infer.py. The structural tests below/above run in BOTH
# modes.
requires_real = pytest.mark.skipif(
    not os.path.exists(COND_NPZ) or not jit_enabled(),
    reason="needs the trained CONDITIONED net (gitignored; regenerate with "
           "experiments/train_profiler.py) AND the JIT (NO_JIT runtime "
           "budget)")


@pytest.fixture(scope="module")
def real_pool():
    cw, _ = load_profiler(COND_NPZ)
    gw, _ = load_profiler(GEN_NPZ)
    return PoolProfiler(cw, pool_ids()), ProfilerLikelihood(gw)


def _match(pid, mates, seed_base, n_hands):
    import gen_population_data as G
    pool = [int(pid)] + [int(x) for x in mates]
    out = []
    for i in range(n_hands):
        rows, table = G.play_and_record(seed_base + i, pool=pool)
        sel = np.asarray(rows["acting_seat"]) == table.index(int(pid))
        out.append((rows["profiler_features"][sel].astype(np.float64),
                    rows["legal_mask"][sel],
                    rows["chosen_card"][sel].astype(np.int64)))
    return out


@requires_real
def test_gate_a_identification_of_a_pool_member(real_pool):
    """GATE (a): watching a POOL member concentrates weight on it.

    Reported both ways (weight > 0.5 AND argmax), because with 12 continuous
    dials a near-twin in the pool can hold the true member's WEIGHT under 0.5
    forever while still being ranked first -- that is pool geometry, not a
    failure to identify.

    Seeds 965000+ are OUTSIDE the training deal range (Task 2 generated
    700000..899999), outside Task 3's held-out eval base (950000) and outside
    this task's measurement bases (980000 / 990000), so the reported N is not
    inflated by deals the CONDITIONED net was trained on.
    """
    pool, _gen = real_pool
    ids = pool.ids
    rng = np.random.default_rng([314159, 5155])
    targets = [ids[int(i)] for i in rng.choice(len(ids), 10, replace=False)]
    n_hands = 12
    w_true = np.zeros(n_hands)
    is_top = np.zeros(n_hands)
    for t, pid in enumerate(targets):
        mates = [i for i in ids if i != pid][:3]
        hands = _match(pid, mates, 965_000 + 100 * t, n_hands)
        mix = SeatMixture(pool)
        ki = ids.index(pid)
        for h in range(n_hands):
            mix.observe(*hands[h])
            w = mix.weights
            w_true[h] += w[ki] / len(targets)
            is_top[h] += (int(w.argmax()) == ki) / len(targets)
    print("\nGATE (a) identification: hands -> mean weight on true member "
          "/ argmax rate")
    for h in range(n_hands):
        print(f"  {h + 1:>2} hands: w_true={w_true[h]:.3f} "
              f"argmax={is_top[h]:.2f}")
    n_half = next((h + 1 for h in range(n_hands) if w_true[h] > 0.5), None)
    print(f"  N at which mean weight on the true member exceeds 0.5: "
          f"{n_half}")
    # The gate itself: identification must be strictly increasing in evidence
    # and must beat chance (1/K) by a wide margin within the match.
    assert w_true[-1] > w_true[0]
    assert w_true[-1] > 10.0 / pool.k
    assert is_top[-1] >= 0.5


@requires_real
def test_gate_b_heldout_benefit_by_hand_three(real_pool):
    """GATE (b): on HELD-OUT personalities, the adapted likelihood beats
    GENERIC by hand 3 -- fitted on hands 1..h, scored on hand h+1."""
    from experiments.measure_adaptation import single_probs  # noqa
    pool, gen = real_pool
    _train, held = train_heldout_ids()
    targets = [int(x) for x in held[:20]]
    n_hands = 7                      # fit up to 6, score up to hand 7
    acc = {k: np.zeros((n_hands - 1, 2)) for k in
           ("GENERIC", "MIXTURE", "BLEND")}
    for t, pid in enumerate(targets):
        mates = [i for i in held if i != pid][:3]
        hands = _match(pid, mates, 990_000 + 100 * t, n_hands)
        mix = SeatMixture(pool)
        for h in range(n_hands - 1):
            mix.observe(*hands[h])
            f, m, c = hands[h + 1]
            if len(c) == 0:
                continue
            pm = pool.member_probs(f, m)
            P_mix = np.einsum("k,nkc->nc", mix.weights, pm)
            P_gen = single_probs(gen.weights, f, m)
            for name, P in (("GENERIC", P_gen), ("MIXTURE", P_mix),
                            ("BLEND", (1 - BLEND_B) * P_mix + BLEND_B * P_gen)):
                p = P[np.arange(len(c)), c]
                assert (p > 0).all()
                acc[name][h] += (np.log(p).sum(), len(c))
    ll = {k: v[:, 0] / v[:, 1] for k, v in acc.items()}
    print("\nGATE (b) held-out trajectory (mean LL on the NEXT hand):")
    for h in range(n_hands - 1):
        print(f"  after {h + 1} hands: GENERIC {ll['GENERIC'][h]:.4f}  "
              f"MIXTURE {ll['MIXTURE'][h]:.4f}  BLEND {ll['BLEND'][h]:.4f}")
    # pre-registered: better by hand 3 (index 2), on average
    assert ll["MIXTURE"][2] >= ll["GENERIC"][2], (
        f"gate (b) FAILED: mixture {ll['MIXTURE'][2]:.4f} < generic "
        f"{ll['GENERIC'][2]:.4f} after 3 hands")
    assert ll["BLEND"][2] >= ll["GENERIC"][2]
