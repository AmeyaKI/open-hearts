"""Phase 6 Task A2: the curve script's load-bearing pure helpers.

Only two things here are logic rather than plumbing, and both decide whether
the headline number means anything:

* `rare_flags` -- the rare-moment slice must be a pure function of
  (deal, rotation), computable before a card is played, or different rows get
  different subsets and the paired diff is meaningless.
* `slice_means` -- restricting to qualifying rotations must average only over
  those rotations, and must DROP (never impute) deals with none.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

import run_entropy_curve as C  # noqa: E402

from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import deal  # noqa: E402


def test_rare_flags_are_a_pure_function_of_the_deal():
    """Same seed, same rotation -> same flags, every time and for every row."""
    for seed in (C.DEAL_SEED_BASE, C.DEAL_SEED_BASE + 7):
        for rot in range(4):
            a = C.rare_flags(deal(np.random.default_rng(seed)), rot)
            b = C.rare_flags(deal(np.random.default_rng(seed)), rot)
            assert a == b


def test_qs_opp_holds_for_exactly_three_of_four_rotations():
    """The bot holds the queen in exactly one seat, so QS-OPP retains 3/4.

    This is the retention fact the report prints beside the slice: the plan's
    literal pre-registration is not a rare event, and pretending otherwise
    would let a null be misread.
    """
    for seed in range(C.DEAL_SEED_BASE, C.DEAL_SEED_BASE + 5):
        state = deal(np.random.default_rng(seed))
        flags = [C.rare_flags(state, r)[0] for r in range(4)]
        assert sum(flags) == 3


def test_qs_exposed_is_a_strict_subset_and_matches_its_definition():
    ace = cards.SPADES * 13 + 12
    king = cards.SPADES * 13 + 11
    for seed in range(C.DEAL_SEED_BASE, C.DEAL_SEED_BASE + 20):
        state = deal(np.random.default_rng(seed))
        for r in range(4):
            qs_opp, exposed = C.rare_flags(state, r)
            hand = state.hands[r]
            want = (not (hand & cards.bit(cards.QUEEN_SPADES))) and bool(
                hand & (cards.bit(ace) | cards.bit(king)))
            assert exposed == want
            if exposed:
                assert qs_opp


def test_slice_means_averages_only_qualifying_rotations():
    rot = np.array([[0.0, 4.0, 8.0, 100.0],
                    [1.0, 2.0, 3.0, 4.0]])
    flags = np.array([[True, True, True, False],
                      [False, False, False, True]])
    vals, keep = C.slice_means(rot, flags)
    assert keep.all()
    assert vals[0] == 4.0        # mean(0, 4, 8), the 100 excluded
    assert vals[1] == 4.0        # the single qualifying rotation


def test_slice_means_drops_deals_with_no_qualifying_rotation():
    rot = np.array([[1.0, 1.0, 1.0, 1.0], [5.0, 5.0, 5.0, 5.0]])
    flags = np.array([[False] * 4, [True] * 4])
    vals, keep = C.slice_means(rot, flags)
    assert keep.tolist() == [False, True]
    assert vals[1] == 5.0
    # the dropped deal is never imputed as a zero into a reported mean
    assert float(np.mean(vals[keep])) == 5.0


def test_rows_and_baseline_match_the_plan():
    """The slim row set the lead specified: no CHOICE-soft, no RI/RIA."""
    assert C.ROW_NAMES == ["honest-FULL", "honest-CHOICE-strict",
                           "profiled-R", "profiled-ORACLE"]
    assert C.BASELINE_ROW == "honest-FULL"
    assert "CHOICE-soft" not in " ".join(C.ROW_NAMES)


def test_the_curve_never_points_at_a_phase_5_artifact():
    for p in (C.generic_npz(), C.conditioned_npz()):
        assert "profiler_train_v2" in p
        assert "models" not in p
