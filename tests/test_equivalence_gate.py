"""THE GATE for equivalence-class card grouping.

Two gates, deliberately separated, because they assert different things.

GATE A -- THE THEOREM (exact, asserted). Matched seeds: every candidate is
evaluated against the SAME outer worlds and the SAME pre-drawn inner seeds.
Under those conditions the members of an equivalence class must produce
BIT-IDENTICAL mean scores, and the grouped choice must equal the ungrouped
choice exactly. That is the whole claim of `search/grouping.py`, and it is an
equality, not a tolerance.

GATE B -- BEHAVIOUR ON THE REAL PATHS (measured and reported, asserted only
where the theorem actually applies). The shipped draw order gives candidate k
and candidate k+1 DIFFERENT inner seeds, so on the ungrouped path two provably
interchangeable cards are scored against different imagined continuations and
get slightly different means. The argmin can therefore land on a different
member of the same class -- or, when two classes are near-tied, on a different
class entirely. Neither is a grouping bug; both are the pre-existing
imagination noise the C0 probe measured. So gate B reports rates rather than
asserting a number nobody has measured, and asserts only the two things that
must hold: grouping never proposes an illegal card, and it never evaluates a
card outside the representative set.
"""
import numpy as np
import pytest

from openhearts.belief.table import BeliefTable, Level
from openhearts.engine import cards, kernel
from openhearts.engine.kernel import jit_enabled
from openhearts.search import grouping
from openhearts.search.honest import HonestSearchPlayer

from test_jit_fused_decision import _play_corpus

pytestmark = pytest.mark.skipif(not jit_enabled(),
                                reason="JIT disabled in this environment")

CFG = (Level.FULL, 10, 5)
N_GAMES = 10


@pytest.fixture(scope="module")
def gate_corpus():
    out = (_play_corpus(*CFG, N_GAMES, 1000)
           + _play_corpus(Level.FULL, 10, 5, N_GAMES, 3000))
    assert len(out) >= 700, f"corpus too small: {len(out)}"
    return out


def _classes_for(view):
    rep_mask, rep_of = grouping.equivalence_classes(
        view.hand, view.legal_moves, grouping.dead_mask_from_view(view))
    reps = cards.cards_in(rep_mask)
    return reps, rep_of


def _matched_seed_means(view, cfg, seed):
    """Every legal card scored against the SAME worlds and the SAME inner
    seeds -- the control that makes the theorem checkable.

    Implemented by calling `honest_decision_kernel` once per candidate with a
    one-element `legal_cards`, handing each call the identical `arrangements`
    and the identical pre-drawn `seeds` array.
    """
    level, n_outer, n_inner = cfg
    rng = np.random.default_rng(seed)
    player = HonestSearchPlayer(level, n_outer, n_inner, rng)
    table = BeliefTable.from_view(view, level)
    arrangements = player._sample(table, n_outer)
    if len(arrangements) * 2 < n_outer:
        return None, None

    legal = cards.cards_in(view.legal_moves)
    all_plays = list(view.history) + list(view.current_trick)
    real_cards = np.zeros(52, dtype=np.int64)
    real_seats = np.zeros(52, dtype=np.int64)
    for k, (s, c) in enumerate(all_plays):
        real_cards[k] = c
        real_seats[k] = s
    tc0 = np.zeros(4, dtype=np.int64)
    ts0 = np.zeros(4, dtype=np.int64)
    for i, (s, c) in enumerate(view.current_trick):
        tc0[i] = c
        ts0[i] = s
    arr = np.array([[int(h) for h in w] for w in arrangements],
                   dtype=np.int64).reshape(len(arrangements), 3)
    opp = np.array([(view.seat + 1 + i) % 4 for i in range(3)],
                   dtype=np.int64)
    # ONE seeds array, reused verbatim by every candidate.
    seeds = np.array([rng.integers(2**63) for _ in range(len(arrangements))],
                     dtype=np.int64)

    means = {}
    for card in legal:
        best, _fb, _fs, _used, avgs, status = kernel.honest_decision_kernel(
            np.int64(view.hand), opp, real_cards, real_seats, len(all_plays),
            tc0, ts0, len(view.current_trick), bool(view.hearts_broken),
            int(view.trick_number), np.array(view.scores, dtype=np.int64),
            np.int64(view.seat), arr, np.array([card], dtype=np.int64),
            int(n_inner), kernel.LEVEL_FULL, True, 200, seeds)
        assert status == kernel._ST_OK, f"kernel status {status}"
        means[card] = float(avgs[0])
    return means, legal


def test_gate_a_matched_seed_theorem(gate_corpus):
    """Class members score bit-identically, and the grouped choice is the
    ungrouped choice -- exactly."""
    n_checked = n_classes = n_multi = 0
    for i, (view, cfg) in enumerate(gate_corpus):
        if i % 3:          # every third decision: this control is ~4x a
            continue       # normal decision (one kernel call per candidate)
        reps, rep_of = _classes_for(view)
        means, legal = _matched_seed_means(view, cfg, 700000 + i)
        if means is None:
            continue
        n_checked += 1
        for rep in reps:
            members = grouping.class_members(rep_of, rep)
            n_classes += 1
            if len(members) > 1:
                n_multi += 1
            for m in members:
                assert means[m] == means[rep], (
                    f"decision {i}: class {[cards.card_name(x) for x in members]}"
                    f" scores {means[m]} vs {means[rep]} under matched seeds")
        # tie-break: strict `<` over ascending order, ties to lowest index
        ungrouped = min(legal, key=lambda c: (means[c], c))
        grouped = min(reps, key=lambda c: (means[c], c))
        assert grouped == ungrouped, (
            f"decision {i}: grouped {cards.card_name(grouped)} vs ungrouped "
            f"{cards.card_name(ungrouped)} under matched seeds")
    assert n_checked >= 200, f"only {n_checked} matched-seed decisions"
    assert n_multi >= 50, f"only {n_multi} multi-card classes exercised"
    print(f"\n[gate A] {n_checked} decisions, {n_classes} classes "
          f"({n_multi} with >1 card): all class means bit-identical, all "
          f"choices identical.")


def _run(view, cfg, seed, group):
    level, n_outer, n_inner = cfg
    rng = np.random.default_rng(seed)
    p = HonestSearchPlayer(level, n_outer, n_inner, rng, fused=True,
                           group_equivalent=group)
    card = p.choose(view)
    return card, p


def test_gate_b_real_paths(gate_corpus):
    """Grouped vs ungrouped on the SAME view and SAME rng seed. Reports
    identical / same-class / cross-class rates and the ungrouped path's
    class-member score spread (the imagination-noise diagnostic)."""
    n = n_same = n_class = 0
    cross = []
    n_cand_before = n_cand_after = 0
    spreads = []
    by_bucket = {}
    for i, (view, cfg) in enumerate(gate_corpus):
        seed = 800000 + i
        legal = cards.cards_in(view.legal_moves)
        reps, rep_of = _classes_for(view)
        u_card, u_p = _run(view, cfg, seed, False)
        g_card, g_p = _run(view, cfg, seed, True)
        if u_p.fallbacks or g_p.fallbacks:
            continue
        n += 1
        n_cand_before += len(legal)
        n_cand_after += len(reps)
        b = by_bucket.setdefault(view.trick_number // 4, [0, 0, 0])
        b[0] += len(legal)
        b[1] += len(reps)
        b[2] += 1

        assert (view.legal_moves >> g_card) & 1, "grouping chose an illegal card"
        assert g_card in reps, "grouping evaluated a non-representative"
        if g_card == u_card:
            n_same += 1
            n_class += 1
        elif rep_of[g_card] == rep_of[u_card]:
            n_class += 1
        else:
            cross.append((i, u_card, g_card))

        # imagination noise between provably equivalent cards, ungrouped path
        if u_p.last_avgs is not None:
            avgs = {c: float(a) for c, a in zip(legal, u_p.last_avgs)}
            for rep in reps:
                members = grouping.class_members(rep_of, rep)
                if len(members) > 1:
                    vals = [avgs[m] for m in members]
                    spreads.append(max(vals) - min(vals))

    print(f"\n[gate B] {n} decisions."
          f"\n  identical card:   {n_same} ({100.0*n_same/n:.1f}%)"
          f"\n  same class:       {n_class} ({100.0*n_class/n:.1f}%)"
          f"\n  cross-class:      {len(cross)} ({100.0*len(cross)/n:.1f}%)"
          f"\n  candidates/decision: {n_cand_before/n:.2f} -> "
          f"{n_cand_after/n:.2f} ({100.0*(1-n_cand_after/n_cand_before):.1f}% "
          f"fewer)")
    for k in sorted(by_bucket):
        b = by_bucket[k]
        print(f"  tricks {4*k}-{4*k+3}: {b[0]/b[2]:.2f} -> {b[1]/b[2]:.2f} "
              f"({b[2]} decisions)")
    if spreads:
        s = np.array(spreads)
        print(f"  ungrouped class-member score spread (imagination noise "
              f"between provably equivalent cards): mean {s.mean():.4f}, "
              f"max {s.max():.4f}, n={len(s)}, zero in "
              f"{100.0*(s == 0).mean():.1f}% of classes")
    assert n >= 700, f"only {n} decisions"


def test_exploiter_model_off_is_still_bitwise_honest_full(gate_corpus):
    """The exploiter's gate with grouping OFF: model-off must remain bitwise
    honest-FULL, card and rng state alike."""
    from openhearts.search.exploiter import ExploiterSearchPlayer
    for i, (view, cfg) in enumerate(gate_corpus[:60]):
        level, n_outer, n_inner = cfg
        r1 = np.random.default_rng(600000 + i)
        r2 = np.random.default_rng(600000 + i)
        a = HonestSearchPlayer(level, n_outer, n_inner, r1)
        b = ExploiterSearchPlayer(level, n_outer, n_inner, r2)
        assert a.choose(view) == b.choose(view)
        assert r1.bit_generator.state == r2.bit_generator.state


def test_exploiter_nested_hook_fires_for_every_representative(gate_corpus):
    """With grouping ON the exploiter must still nest a champion at the first
    imagined ply after EVERY candidate it evaluates -- and the candidates are
    exactly the representatives."""
    from openhearts.search.exploiter import (ExploiterSearchPlayer,
                                             champion_model_factory)
    n_checked = n_grouped = n_full = 0
    for i, (view, cfg) in enumerate(gate_corpus[:40]):
        level, n_outer, n_inner = cfg
        seats = [s for s in range(4) if s != view.seat]
        p = ExploiterSearchPlayer(
            level, n_outer, n_inner, np.random.default_rng(610000 + i),
            champion_model=champion_model_factory(level, 4, 2),
            champion_seats=seats, model_seed=7,
            measuring_instrument=True, group_equivalent=True)
        before = p.nested_calls
        p.choose(view)
        reps, _rep_of = _classes_for(view)
        assert p.last_candidates == reps
        # `pred_counts` is keyed by the candidate we played; it records the
        # ply IMMEDIATELY after it, which exists only when that ply belongs to
        # a champion seat facing a real choice. So it is a SUBSET of the
        # representatives -- and never contains a non-representative, which is
        # the property grouping could break.
        assert set(p.pred_counts) <= set(reps), (
            f"decision {i}: nested hook fired for a non-representative: "
            f"{sorted(set(p.pred_counts) - set(reps))}")
        assert p.nested_calls > before, f"decision {i}: hook never fired"
        n_full += set(p.pred_counts) == set(reps)
        n_checked += 1
        if len(reps) < len(cards.cards_in(view.legal_moves)):
            n_grouped += 1
    assert n_checked == 40
    assert n_grouped >= 10, f"only {n_grouped} decisions actually grouped"
    assert n_full >= 20, (
        f"only {n_full}/40 decisions had the hook fire for EVERY "
        f"representative")
