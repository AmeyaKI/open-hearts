"""Gates for the Phase-7A expert population (PHASE7_PLAN.md 7A-1 spec §3-§4,
G5 plus the schema/seed/split/wall contracts).

Pre-registered floors (frozen 2026-09-05, before any measurement):
  (i)   mean pairwise greedy disagreement over 100 seeded train pairs >= 0.12
  (ii)  every train expert vs plain HeuristicPlayer >= 0.10; mean >= 0.20
  (iii) zero behavioral duplicates among the train pairs measured
  (iv)  expert mirror (4 copies, independent rngs, 300 deals x 4 rotations)
        brackets 6.5 for REFERENCE_PARAMS and train ids 11, 12, 13
Measured with the fixed-tie (greedy) choice, the same reason Phase 5 used
argmax: sampling noise must neither manufacture nor hide a difference.
States come from ONE reference driver (the reference config at all four
seats), so neither comparand biases the state distribution.
"""
import dataclasses
import math

import numpy as np
import pytest

from openhearts.engine import cards
from openhearts.engine.game import deal
from openhearts.eval.harness import rotated_match
from openhearts.eval.stats import bootstrap_ci
from openhearts.players.expert import ExpertPlayer, greedy_pick, action_distribution
from openhearts.players.expert_population import (
    DOMAIN_POPULATION, DOMAIN_DEALS, DOMAIN_POLICY_TIES, MASTER_SEED_EXPERT_V1,
    ExpertParams, OPTIONAL_WEIGHTS, REFERENCE_PARAMS, all_optional_off,
    check_coherence, derive_seed, dump_params, heldout_ids, is_heldout,
    make_expert_population, record_key, sample_expert, train_ids,
)
from openhearts.players.heuristic import HeuristicPlayer

DIVERGENCE_FLOOR = 0.12
HEURISTIC_FLOOR_EACH = 0.10
HEURISTIC_FLOOR_MEAN = 0.20

FIELD_ORDER = (
    "expert_id", "tie_mode", "w_Q1", "queen_hunt_until", "hunt_min_spades",
    "w_guard", "guard_target", "w_F4", "w_X1", "w_F5", "w_L2", "w_D2",
    "void_window", "w_exit", "exit_count", "w_L4", "w_L5",
    "void_lead_threshold", "guarded_spade_first", "w_exposed", "w_disposal",
    "w_F1", "w_F1_spades", "w_F1_hearts", "w_F2", "w_H4", "ace_release_cost",
    "w_H5", "w_L1", "escape_boost", "high_hand_mean", "small_rank",
)


# ---------------------------------------------------------------- corpus

def _multi_legal_views(deal_seed, params=REFERENCE_PARAMS):
    st = deal(np.random.default_rng(deal_seed))
    drv = ExpertPlayer(np.random.default_rng(0), params)
    out = []
    while not st.is_over():
        v = st.view_for(st.to_play)
        if bin(v.legal_moves).count("1") > 1:
            out.append(v)
        st.play(drv.choose(v))
    return out


_CORPUS = None


def corpus():
    global _CORPUS
    if _CORPUS is None:
        views = []
        for k in range(50):
            views += _multi_legal_views(70000 + k)
        _CORPUS = views
    return _CORPUS


def _disagreement(views, pa, pb):
    return sum(1 for v in views if greedy_pick(v, pa) != greedy_pick(v, pb)) / len(views)


# ---------------------------------------------------------------- contracts

def test_derive_seed_is_pinned_and_domain_separated():
    assert derive_seed(1, 7070707, 1) == 0x49df98b52b9bbfc3
    assert derive_seed(2, 0) == 0x64684c4f0fd784b4
    assert derive_seed(DOMAIN_POPULATION, 5) != derive_seed(DOMAIN_DEALS, 5)
    assert derive_seed(DOMAIN_POLICY_TIES, 5, 6) != derive_seed(DOMAIN_POLICY_TIES, 6, 5)
    assert 0 <= derive_seed(3, -1, 2 ** 70) < 2 ** 64


def test_schema_field_order_is_the_frozen_draw_order():
    names = tuple(f.name for f in dataclasses.fields(ExpertParams))
    assert names == FIELD_ORDER
    with pytest.raises(dataclasses.FrozenInstanceError):
        REFERENCE_PARAMS.w_L1 = 0.0


def test_split_is_200_50_by_index_and_deterministic():
    tr, he = make_expert_population()
    assert tr == list(range(1, 201)) and he == list(range(201, 251))
    assert all(not is_heldout(i) for i in tr) and all(is_heldout(i) for i in he)
    assert not is_heldout(0) and not is_heldout(251)


def test_records_are_deterministic_distinct_and_coherent():
    ids = train_ids() + heldout_ids()
    recs = [sample_expert(i) for i in ids]
    assert [sample_expert(i) for i in ids[:5]] == recs[:5]
    assert len({record_key(r) for r in recs}) == len(recs)
    ref = record_key(REFERENCE_PARAMS)
    assert all(record_key(r) != ref for r in recs)
    for r in recs:
        check_coherence(r)
        assert sum(1 for w in OPTIONAL_WEIGHTS if getattr(r, w) > 0) >= 5


def test_reference_config_matches_the_catalogue():
    r = REFERENCE_PARAMS
    assert all(getattr(r, w) == 1.0 for w in OPTIONAL_WEIGHTS)
    assert (r.queen_hunt_until, r.hunt_min_spades, r.guard_target,
            r.void_lead_threshold, r.void_window, r.exit_count,
            r.ace_release_cost, r.guarded_spade_first, r.high_hand_mean,
            r.small_rank, r.escape_boost, r.tie_mode) == (
        5, 3, 1, 2, 8, 1, 4, False, 8.0, 4, 0.5, "fixed")
    off = all_optional_off()
    assert all(getattr(off, w) == 0.0 for w in OPTIONAL_WEIGHTS)
    with pytest.raises(AssertionError):
        check_coherence(off)


def test_heldout_wall_dump_refused_and_repr_withheld():
    assert len(dump_params([1, 2, 3])) == 3
    with pytest.raises(PermissionError):
        dump_params([1, 201])
    r = sample_expert(250)
    assert "withheld" in repr(r) and "w_" not in repr(r) and "withheld" in str(r)
    assert "w_Q1" in repr(sample_expert(1))


def test_all_optional_off_plays_legally_everywhere():
    off = all_optional_off()
    for seed in range(20):
        st = deal(np.random.default_rng(5000 + seed))
        ps = [ExpertPlayer(np.random.default_rng(seed), off) for _ in range(4)]
        while not st.is_over():
            v = st.view_for(st.to_play)
            c = ps[st.to_play].choose(v)
            assert v.legal_moves & cards.bit(c)
            st.play(c)
        assert sum(st.scores) == 26


# ---------------------------------------------------------------- G5 floors

def test_divergence_between_train_pairs():
    views = corpus()
    rng = np.random.default_rng(2026)
    tr = train_ids()
    rates = []
    for _ in range(100):
        a, b = rng.choice(tr, size=2, replace=False)
        rates.append(_disagreement(views, sample_expert(int(a)), sample_expert(int(b))))
    mean = float(np.mean(rates))
    print(f"\n[G5] pairwise divergence mean={mean:.3f} min={min(rates):.3f} "
          f"max={max(rates):.3f} over {len(views)} views")
    assert min(rates) > 0.0, "behavioral duplicate found"
    assert mean >= DIVERGENCE_FLOOR, f"pair divergence {mean:.3f}"


def test_divergence_self_baseline_is_zero():
    views = corpus()[:300]
    assert _disagreement(views, sample_expert(4), sample_expert(4)) == 0.0


class _HG:
    def __init__(self):
        self.h = HeuristicPlayer()


def test_every_train_expert_diverges_from_the_heuristic():
    views = corpus()
    h = HeuristicPlayer()
    rates = []
    for i in train_ids():
        p = sample_expert(i)
        rates.append(sum(1 for v in views if greedy_pick(v, p) != h.choose(v)) / len(views))
    print(f"\n[G5] vs-heuristic mean={np.mean(rates):.3f} min={min(rates):.3f} "
          f"(id {train_ids()[int(np.argmin(rates))]})")
    assert min(rates) >= HEURISTIC_FLOOR_EACH, f"min vs-heuristic {min(rates):.3f}"
    assert float(np.mean(rates)) >= HEURISTIC_FLOOR_MEAN


@pytest.mark.parametrize("pid", [0, 11, 12, 13])
def test_expert_mirror_brackets_six_point_five(pid):
    params = REFERENCE_PARAMS if pid == 0 else sample_expert(pid)
    counter = [0]

    def factory():
        counter[0] += 1
        return ExpertPlayer(np.random.default_rng(
            derive_seed(DOMAIN_POLICY_TIES, 900000, pid, counter[0])), params)

    per_deal = rotated_match(range(100000, 100300), factory, factory)
    mean, lo, hi = bootstrap_ci(per_deal, rng=np.random.default_rng(7))
    assert counter[0] == 300 * 4 * 4, "alarm must count its own firings"
    assert lo <= 6.5 <= hi, f"mirror mean {mean:.3f} CI ({lo:.2f}, {hi:.2f}) excludes 6.5"


def test_entropy_report_and_fixed_tie_zero():
    """Reported, not gated (expectations pre-stated in the plan): within-
    expert entropy is exactly 0 nats under fixed ties; between-expert entropy
    of the train mixture is where the gym's width lives."""
    views = corpus()[::4]
    tr = train_ids()
    within = {"fixed": [], "seeded_uniform": []}
    mixture_h, contested_h = [], []
    for v in views:
        agg = {}
        for i in tr:
            p = sample_expert(i)
            cs, ps = action_distribution(v, p)
            h = -sum(x * math.log(x) for x in ps if x > 0)
            within[p.tie_mode].append(h)
            for c, x in zip(cs, ps):
                agg[c] = agg.get(c, 0.0) + x / len(tr)
        hm = -sum(x * math.log(x) for x in agg.values() if x > 0)
        mixture_h.append(hm)
        if len(agg) > 1:
            contested_h.append(hm)
    assert within["fixed"] and max(within["fixed"]) == 0.0
    print(f"\n[G5] within-expert entropy: fixed={np.mean(within['fixed']):.4f} nats, "
          f"seeded_uniform={np.mean(within['seeded_uniform']):.4f} nats; "
          f"between-expert mixture={np.mean(mixture_h):.3f} nats over {len(views)} views, "
          f"{np.mean(contested_h):.3f} on {len(contested_h)} contested views "
          f"({len(contested_h)/len(views):.1%})")
