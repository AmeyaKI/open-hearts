"""C-bench 1: OpenSpiel adapter tests (experiments/cbench/adapter.py).

Requires pyspiel; skips cleanly in the project .venv (which does not, and
must not, depend on OpenSpiel). Run these from the isolated scratch venv
that has open_spiel + openhearts installed, e.g.:

    <scratch-venv>/bin/python -m pytest tests/test_cbench.py -q
"""
import os
import random
import sys

import pytest

pytest.importorskip("pyspiel")

import pyspiel  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from cbench import adapter  # noqa: E402
from openhearts.engine import cards  # noqa: E402


def test_card_bijection_round_trip_all_52():
    for card_ours in range(52):
        card_os = adapter.ours_to_os(card_ours)
        assert 0 <= card_os < 52
        assert adapter.os_to_ours(card_os) == card_ours
    for card_os in range(52):
        card_ours = adapter.os_to_ours(card_os)
        assert 0 <= card_ours < 52
        assert adapter.ours_to_os(card_ours) == card_os


def test_card_bijection_known_points():
    assert adapter.os_to_ours(43) == cards.QUEEN_SPADES  # QS
    assert adapter.ours_to_os(cards.QUEEN_SPADES) == 43
    assert adapter.os_to_ours(0) == cards.TWO_CLUBS  # 2C in both encodings
    assert adapter.ours_to_os(cards.TWO_CLUBS) == 0


def test_card_bijection_matches_action_to_string():
    """Cross-check against OpenSpiel's own action_to_string labels."""
    g = pyspiel.load_game(adapter.GAME_STRING)
    s = g.new_initial_state()
    s.apply_action(0)  # pass direction (inert)
    name_to_os = {}
    for a in range(52):
        name_to_os[s.action_to_string(a)] = a
    # Spot-check a handful against our own card_name().
    for card_ours in [cards.TWO_CLUBS, cards.QUEEN_SPADES, 12, 25, 51]:
        name_ours = cards.card_name(card_ours)  # e.g. "Ac", "Th"
        # OpenSpiel names are like "QS", "2C" (rank+suit, uppercase suit).
        name_os = name_ours[0].upper() + name_ours[1].upper()
        expected_os_id = name_to_os[name_os]
        assert adapter.ours_to_os(card_ours) == expected_os_id


def _play_random_game(seed, trial_seed):
    random.seed(trial_seed)
    g = pyspiel.load_game(adapter.GAME_STRING)
    s = g.new_initial_state()
    mirror = adapter.force_deal(s, seed=seed)
    while not s.is_terminal():
        adapter.assert_legal_agreement(s, mirror)  # the tripwire, exercised live
        legal = s.legal_actions()
        a = random.choice(legal)
        card_ours = adapter.os_to_ours(a)
        adapter.apply_both(s, mirror, card_ours)
    return s, mirror


@pytest.mark.parametrize("trial", range(200))
def test_legal_move_agreement_random_games(trial):
    """The live rules-alignment tripwire: legal actions must agree at every
    decision point across ~200 random-policy games. Any disagreement raises
    inside adapter.assert_legal_agreement (not silently skipped)."""
    s, mirror = _play_random_game(seed=100000 + trial, trial_seed=trial)
    assert s.is_terminal()
    assert mirror.is_over()


@pytest.mark.parametrize("trial", range(50))
def test_scoring_agreement_no_moon(trial):
    """Our history-rescoring must equal OpenSpiel's own returns() whenever
    no moon was shot (26 - points transform). Moon-shoot deals are a known,
    documented divergence (RULES_ALIGNMENT.md sec 3) -- not compared here,
    but detected and reported rather than silently passed."""
    s, mirror = _play_random_game(seed=200000 + trial, trial_seed=1000 + trial)
    mscores = adapter.rescore(mirror)
    assert sum(mscores) == 26
    moon = any(v == 26 for v in mscores)
    osret = s.returns()
    if moon:
        # Documented divergence: OpenSpiel applies its own moon-shoot
        # rescoring (shooter -> 0, others -> +26); we never do. Just assert
        # our own tally is sane and move on -- comparing against osret here
        # would be asserting a known false equality.
        assert mscores.count(26) == 1
    else:
        expected = [26 - p for p in mscores]
        assert osret == expected, (
            f"scoring mismatch on a non-moon deal: our rescoring {mscores} "
            f"-> expected returns {expected}, openspiel gave {osret}"
        )
