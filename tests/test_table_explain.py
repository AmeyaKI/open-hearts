"""Tests for apps/table/explain.py: every number in a sentence must be
recomputable from the view and the belief table."""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apps.table.explain import pretty, sentences  # noqa: E402
from openhearts.belief.table import BeliefTable, Level  # noqa: E402
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import deal  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402

NAMES = ["South", "West", "North", "East"]
SUITS = ["clubs", "diamonds", "spades", "hearts"]


def state_after(seed, plays):
    st = deal(np.random.default_rng(seed))
    p = HeuristicPlayer()
    for _ in range(plays):
        st.play(p.choose(st.view_for(st.to_play)))
    return st


def view_table(seed, plays):
    st = state_after(seed, plays)
    view = st.view_for(st.to_play)
    return view, BeliefTable.from_view(view, Level.FULL)


def scan(predicate, plays_range=range(0, 48, 3), seeds=range(60)):
    for seed in seeds:
        for n in plays_range:
            view, table = view_table(seed, n)
            if predicate(view, table):
                return view, table
    pytest.fail("no state found matching predicate")


def first_reveal(view, seat, s):
    """Independent scan: 1-based trick where seat first failed to follow s."""
    plays = list(view.history) + list(view.current_trick)
    for t in range(0, len(plays), 4):
        trick = plays[t:t + 4]
        if cards.suit(trick[0][1]) != s:
            continue
        for who, c in trick[1:]:
            if who == seat and cards.suit(c) != s:
                return t // 4 + 1
    return None


def queen_unseen_split(view, table):
    col = table.probs[:, 36]
    return (table.unseen_mask & cards.bit(36)
            and (col > 0).sum() >= 2)


def test_queen_percentages_match_probs():
    view, table = scan(queen_unseen_split)
    out = sentences(view, table, None, None, NAMES)
    q = [s for s in out if s.startswith("Q♠ is ")]
    assert len(q) == 1
    found = dict((name, int(pct)) for pct, name in
                 re.findall(r"(\d+)% with (\w+)", q[0]))
    expected = {NAMES[seat]: round(100 * float(table.probs[i, 36]))
                for i, seat in enumerate(table.opponent_seats)
                if table.probs[i, 36] > 0}
    assert found == expected


def test_queen_must_hold_when_single_holder():
    view, table = scan(lambda v, t: t.unseen_mask & cards.bit(36)
                       and (t.probs[:, 36] > 0).sum() == 1,
                       plays_range=range(0, 48, 1), seeds=range(200))
    i = int(np.flatnonzero(table.probs[:, 36] > 0)[0])
    out = sentences(view, table, None, None, NAMES)
    assert f"{NAMES[table.opponent_seats[i]]} must hold Q♠." in out


def test_you_hold_queen():
    view, table = scan(lambda v, t: v.hand & cards.bit(36))
    out = sentences(view, table, None, None, NAMES)
    assert "You hold Q♠." in out
    assert not any(s.startswith("Q♠ is") for s in out)


def test_queen_played_no_sentence():
    view, table = scan(lambda v, t: not (v.hand & cards.bit(36))
                       and not (t.unseen_mask & cards.bit(36)))
    out = sentences(view, table, None, None, NAMES)
    assert not any("Q♠" in s for s in out)


def test_void_sentence_names_right_seat_suit_and_trick():
    checked = 0
    for seed in range(40):
        for n in range(4, 48, 5):
            view, table = view_table(seed, n)
            out = sentences(view, table, None, None, NAMES)
            void_lines = [s for s in out if " showed void in " in s]
            has_void = any(table.voids)
            assert bool(void_lines) == has_void
            for line in void_lines:
                name = line.split(" showed void in ")[0]
                seat = NAMES.index(name)
                i = table.opponent_seats.index(seat)
                assert table.voids[i], line
                body = line.split(" showed void in ")[1].split(", so ")[0]
                for s in table.voids[i]:
                    assert SUITS[s] in body
                    t = first_reveal(view, seat, s)
                    assert t is not None
                    assert f"trick {t}" in body
                for s in range(4):
                    if s not in table.voids[i]:
                        assert SUITS[s] not in body
                checked += 1
    assert checked > 10


def test_void_sentences_most_recent_first_and_at_most_two():
    for seed in range(40):
        view, table = view_table(seed, 40)
        out = sentences(view, table, None, None, NAMES)
        void_lines = [s for s in out if " showed void in " in s]
        assert len(void_lines) <= 2
        latest = [max(int(x) for x in re.findall(
            r"trick (\d+)", s.split(", so ")[0])) for s in void_lines]
        assert latest == sorted(latest, reverse=True)


def test_no_void_named_when_voids_empty():
    view, table = scan(lambda v, t: not any(t.voids) and len(v.history) > 0)
    out = sentences(view, table, [0, 1], [1.0, 2.0], NAMES)
    assert not any("void" in s for s in out)


def test_void_clause_counts_unseen_suit_cards():
    view, table = scan(lambda v, t: any(t.voids), plays_range=range(8, 40, 4))
    out = sentences(view, table, None, None, NAMES)
    for line in out:
        for n, noun in re.findall(r"the (\d+) (\w+) still out", line):
            s = [x for x in range(4)
                 if noun in (SUITS[x], SUITS[x][:-1])][0]
            count = bin(table.unseen_mask & cards.SUIT_MASK[s]).count("1")
            assert int(n) == count


def test_hearts_count_matches_unseen_hearts():
    def cond(v, t):
        played = any(cards.suit(c) == 3
                     for _, c in list(v.history) + list(v.current_trick))
        return (played or v.hearts_broken) and \
            t.unseen_mask & cards.SUIT_MASK[3]
    view, table = scan(cond, plays_range=range(12, 48, 4))
    unseen_hearts = [c for c in range(39, 52)
                     if not view.hand & (1 << c)
                     and c not in {c2 for _, c2 in
                                   list(view.history) +
                                   list(view.current_trick)}]
    out = sentences(view, table, None, None, NAMES)
    h = [s for s in out if "still out;" in s]
    assert len(h) == 1
    assert int(h[0].split()[0]) == len(unseen_hearts)
    assert pretty(max(unseen_hearts)) in h[0]


def test_no_hearts_sentence_before_any_heart():
    view, table = view_table(0, 0)
    out = sentences(view, table, None, None, NAMES)
    assert not any("hearts are still out" in s for s in out)


def test_comparison_gap():
    view, table = view_table(1, 0)
    qs, ah, two_c = 36, 51, 0
    out = sentences(view, table, [qs, ah, two_c], [3.0, 1.3, 4.5], NAMES)
    assert out[-1] == ("A♥ averages 1.3 points over the imagined finishes, "
                       "1.7 fewer than Q♠.")
    out = sentences(view, table, [10, 9], [2.0, 0.5], NAMES)
    assert out[-1] == ("J♣ averages 0.5 points over the imagined finishes, "
                       "1.5 fewer than Q♣.")


def test_comparison_coin_flip():
    view, table = view_table(1, 0)
    out = sentences(view, table, [8, 12], [2.02, 2.06], NAMES)
    assert out[-1] == "10♣ and A♣ are a coin flip (2.0 vs 2.1 points)."


def test_comparison_single_candidate():
    view, table = scan(lambda v, t: bin(v.legal_moves).count("1") == 1,
                       plays_range=range(1, 48, 2))
    only = cards.cards_in(view.legal_moves)[0]
    out = sentences(view, table, [only], [3.0], NAMES)
    assert out[-1] == f"{pretty(only)} is the only legal play."


def test_no_comparison_when_candidates_none():
    view, table = view_table(3, 20)
    out = sentences(view, table, None, None, NAMES)
    assert not any("averages" in s or "coin flip" in s or "only legal" in s
                   for s in out)


def test_all_sentences_wellformed_and_capped():
    for seed in range(15):
        for n in range(0, 52, 4):
            view, table = view_table(seed, n)
            legal = cards.cards_in(view.legal_moves)
            avgs = [float(k) * 0.7 for k in range(len(legal))]
            for cand, a in ((legal, avgs), (None, None)):
                out = sentences(view, table, cand, a, NAMES)
                assert isinstance(out, list)
                assert len(out) <= 4
                for s in out:
                    assert isinstance(s, str) and s.strip()
                    assert s.endswith(".")


def test_pretty_names():
    assert pretty(36) == "Q♠"
    assert pretty(47) == "10♥"
    assert pretty(0) == "2♣"
