"""Tests for apps/table/explain.py: every number in a sentence must be
recomputable from the view and the belief table."""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apps.table.explain import (  # noqa: E402
    MAX_SENTENCES, _follow_sentence, _hearts_broken_sentence,
    _hearts_sentence, _lead_sentences, _points_left_sentence,
    _queen_danger_sentence, _your_voids_sentence, pretty, sentences)
from openhearts.belief.table import BeliefTable, Level  # noqa: E402
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import deal  # noqa: E402
from openhearts.engine.state import GameState  # noqa: E402
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
    # The hearts-out line is lowest priority, so in a busy state it may
    # be trimmed; that is only allowed when the output is full.
    out = sentences(view, table, None, None, NAMES)
    h = [s for s in out if "still out;" in s]
    assert len(h) == 1 or len(out) == MAX_SENTENCES
    line = _hearts_sentence(view, table)
    assert line is not None
    assert h in ([], [line])
    assert int(line.split()[0]) == len(unseen_hearts)
    assert pretty(max(unseen_hearts)) in line


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
                assert len(out) <= MAX_SENTENCES
                for s in out:
                    assert isinstance(s, str) and s.strip()
                    assert s.endswith(".")


def test_pretty_names():
    assert pretty(36) == "Q♠"
    assert pretty(47) == "10♥"
    assert pretty(0) == "2♣"


# ---------------------------------------------------------------- new templates

def C(name):
    """'5c' -> card int, '10' written as 'T'."""
    return "23456789TJQKA".index(name[0]) + 13 * "cdsh".index(name[1])


def mask(*names):
    m = 0
    for n in names:
        m |= 1 << C(n)
    return m


def hand_view(hand, current_trick=(), history=(), trick_number=1,
              hearts_broken=False, seat=0):
    """A hand-built view for the observer at `seat` (helpers that read
    only the view)."""
    hands = [0, 0, 0, 0]
    hands[seat] = hand
    st = GameState(hands=hands, history=list(history),
                   current_trick=list(current_trick),
                   hearts_broken=hearts_broken, trick_number=trick_number,
                   to_play=seat)
    return st.view_for(seat)


def all_plays(view):
    return list(view.history) + list(view.current_trick)


def suits_held(view):
    return {cards.suit(c) for c in cards.cards_in(view.hand)}


# A. your voids

def test_your_voids_lists_suits_and_throw_word():
    checked = 0
    for seed in range(30):
        for n in range(8, 48, 4):
            view, table = view_table(seed, n)
            line = _your_voids_sentence(view)
            void = [s for s in range(3) if s not in suits_held(view)]
            if not void or not view.hand:
                assert line is None
                continue
            assert line.startswith("You have no ")
            head = line.split(" left, so ")[0]
            for s in range(4):
                assert (SUITS[s] in head) == (s in void)
            has_h = bool(view.hand & cards.SUIT_MASK[3])
            has_q = bool(view.hand & (1 << 36))
            if not (has_h or has_q):
                assert line.endswith("you cannot be forced to follow there.")
            else:
                word = ("a point card" if has_h and has_q else
                        "a heart" if has_h else "Q♠")
                assert f"you can throw {word} on any " in line
            checked += 1
    assert checked > 5


def test_your_voids_exact_strings():
    v = hand_view(mask("2c", "5h"), trick_number=3)
    assert _your_voids_sentence(v) == ("You have no diamonds or spades "
                                       "left, so you can throw a heart on "
                                       "any diamond or spade trick.")
    v = hand_view(mask("2c", "Qs", "5h"), trick_number=3)
    assert _your_voids_sentence(v) == ("You have no diamonds left, so you "
                                       "can throw a point card on any "
                                       "diamond trick.")
    v = hand_view(mask("2c", "3s"), trick_number=3)
    assert _your_voids_sentence(v) == ("You have no diamonds left, so you "
                                       "cannot be forced to follow there.")


def test_your_voids_first_trick_no_throw_claim():
    # Points may not be thrown on the first trick.
    v = hand_view(mask("2c", "3s", "5h"), trick_number=0)
    assert _your_voids_sentence(v) == ("You have no diamonds left, so you "
                                       "cannot be forced to follow there.")


def test_your_voids_negative():
    v = hand_view(mask("2c", "3d", "4s"), trick_number=2)
    assert _your_voids_sentence(v) is None  # hearts void is not listed
    v = hand_view(0, trick_number=12)
    assert _your_voids_sentence(v) is None
    view, table = view_table(0, 0)
    out = sentences(view, table, None, None, NAMES)
    if all(s in suits_held(view) for s in range(3)):
        assert not any(s.startswith("You have no ") for s in out)


# B. following: duck or win

def test_follow_can_duck_exact():
    v = hand_view(mask("5c", "9c", "Kc", "2d"),
                  current_trick=[(1, C("Tc"))], trick_number=2)
    assert _follow_sentence(v) == ("Clubs were led; the 10♣ is winning so "
                                   "far and you can duck with 5♣ (or 9♣).")


def test_follow_cannot_duck_with_points_exact():
    v = hand_view(mask("5c", "Kc"),
                  current_trick=[(2, C("4c")), (3, C("2h"))],
                  trick_number=4)
    assert _follow_sentence(v) == (
        "Clubs were led; the 4♣ is winning so far and you cannot duck "
        "\u2014 every club you hold beats the 4♣. There is 1 point on "
        "this trick.")
    v = hand_view(mask("As", "3c"),
                  current_trick=[(1, C("2s")), (2, C("Qs")), (3, C("9h"))],
                  trick_number=4)
    assert _follow_sentence(v).endswith(
        "beats the Q♠. There are 14 points on this trick.")


def test_follow_matches_independent_recount():
    checked = 0
    for seed in range(40):
        for n in range(1, 48, 2):
            view, table = view_table(seed, n)
            line = _follow_sentence(view)
            trick = list(view.current_trick)
            led = cards.suit(trick[0][1]) if trick else None
            if not trick or led not in suits_held(view):
                assert line is None
                continue
            winner = max((c for _, c in trick if cards.suit(c) == led),
                         key=cards.rank)
            assert f"the {pretty(winner)} is winning" in line
            assert line.startswith(SUITS[led].capitalize() + " were led;")
            mine = [c for c in cards.cards_in(view.hand)
                    if cards.suit(c) == led]
            ducks = [c for c in mine if cards.rank(c) < cards.rank(winner)]
            if ducks:
                assert f"can duck with {pretty(ducks[0])}" in line
                for c in ducks[1:]:
                    assert pretty(c) in line
            else:
                assert "cannot duck" in line
            pts = sum(1 for _, c in trick if c >= 39) + \
                (13 if any(c == 36 for _, c in trick) else 0)
            if pts:
                assert re.search(rf"\b{pts} points? on this trick", line)
            else:
                assert "on this trick" not in line
            checked += 1
    assert checked > 10


def test_follow_negative_when_leading_or_void():
    assert _follow_sentence(hand_view(mask("5c"), trick_number=2)) is None
    v = hand_view(mask("5d", "6h"), current_trick=[(1, C("Tc"))],
                  trick_number=2)
    assert _follow_sentence(v) is None


# C. leading: risky and safe suits

def leading_with_void(v, t):
    return not v.current_trick and v.trick_number >= 1 and any(t.voids)


def test_lead_sentences_match_voids():
    checked = 0
    for seed in range(60):
        for n in range(4, 48, 1):
            view, table = view_table(seed, n)
            if not leading_with_void(view, table):
                continue
            out = _lead_sentences(view, table, NAMES)
            lead_suits = {cards.suit(c)
                          for c in cards.cards_in(view.legal_moves)}
            if not view.hearts_broken:
                lead_suits.discard(3)
            safe = [s for s in sorted(lead_suits)
                    if not any(s in v for v in table.voids)]
            safe_lines = [s for s in out if "safe to lead" in s]
            assert bool(safe_lines) == bool(safe)
            if safe:
                head = safe_lines[0].split(" are safe")[0].lower()
                for s in range(4):
                    assert (SUITS[s] in head) == (s in safe)
            risky = " ".join(s for s in out if "is risky" in s)
            for s in sorted(lead_suits):
                if f"Leading {SUITS[s]} is risky" not in risky:
                    continue
                part = risky.split(f"Leading {SUITS[s]} is risky: ")[1]
                part = part.split(" void and")[0]
                for i, seat in enumerate(table.opponent_seats):
                    if NAMES[seat] in part:
                        assert s in table.voids[i]
            checked += 1
    assert checked > 5


def test_lead_risky_found_and_named():
    view, table = scan(
        lambda v, t: leading_with_void(v, t) and any(
            "is risky" in s for s in _lead_sentences(v, t, NAMES)),
        plays_range=range(4, 48, 1), seeds=range(200))
    line = [s for s in _lead_sentences(view, table, NAMES)
            if "is risky" in s][0]
    m = re.match(r"Leading (\w+) is risky: (.+?) (is|are) void and can "
                 r"dump (a heart|Q♠|a point card)\.", line)
    assert m
    s = SUITS.index(m.group(1))
    for i, seat in enumerate(table.opponent_seats):
        if NAMES[seat] in m.group(2):
            assert s in table.voids[i]


def test_lead_negative():
    # following, first trick, or no opponent void: nothing said
    view, table = scan(lambda v, t: bool(v.current_trick) and any(t.voids),
                       plays_range=range(5, 48, 2))
    assert _lead_sentences(view, table, NAMES) == []
    view, table = view_table(0, 0)
    assert _lead_sentences(view, table, NAMES) == []
    view, table = scan(lambda v, t: not v.current_trick and v.trick_number
                       >= 1 and not any(t.voids), plays_range=range(4, 48, 4))
    assert _lead_sentences(view, table, NAMES) == []


# D. queen danger

def test_queen_danger_high_spade():
    view, table = scan(lambda v, t: t.unseen_mask & (1 << 36) and
                       v.hand & ((1 << 37) | (1 << 38)))
    line = _queen_danger_sentence(view, table)
    high = [pretty(c) for c in (38, 37) if view.hand & (1 << c)]
    assert line.startswith("You hold " + " and ".join(high) + " above")
    assert line.endswith("could drop Q♠ on you.")
    assert line in sentences(view, table, None, None, NAMES)


def test_queen_danger_holding_queen():
    def cond(v, t):
        spade_led = any(cards.suit(c) == 2 for _, c in all_plays(v)[::4])
        void = any(not v.hand & cards.SUIT_MASK[s] for s in range(4))
        return v.hand & (1 << 36) and spade_led and (void or v.hearts_broken)
    view, table = scan(cond, plays_range=range(4, 48, 2), seeds=range(200))
    assert _queen_danger_sentence(view, table) == (
        "You still hold Q♠; it is safest to dump it on a trick someone "
        "else is winning.")


def test_queen_danger_negative():
    # holds Q♠, spades led, but no void and hearts unbroken
    led_spade = [(1, C("2s")), (2, C("3s")), (3, C("4s")), (0, C("5s"))]
    v = hand_view(mask("Qs", "3c", "3d", "3h"), history=led_spade,
                  trick_number=1)
    t = BeliefTable.from_view(v, Level.VOIDS)
    assert _queen_danger_sentence(v, t) is None
    # Q♠ already played: nothing
    view, table = scan(lambda v, t: not (v.hand & (1 << 36)) and
                       not (t.unseen_mask & (1 << 36)))
    assert _queen_danger_sentence(view, table) is None
    # Q♠ unseen but no A♠/K♠ held
    view, table = scan(lambda v, t: t.unseen_mask & (1 << 36) and not
                       v.hand & ((1 << 37) | (1 << 38)))
    assert _queen_danger_sentence(view, table) is None


# E. hearts broken

def first_heart(view):
    for k, (seat, c) in enumerate(all_plays(view)):
        if c >= 39:
            return k // 4 + 1, seat
    return None


def test_hearts_broken_names_trick_and_seat():
    seen_you = seen_other = False
    for seed in range(60):
        for n in range(4, 48, 3):
            view, table = view_table(seed, n)
            line = _hearts_broken_sentence(view, NAMES)
            fh = first_heart(view)
            if fh is None:
                assert line is None
                continue
            t, seat = fh
            who = "you" if seat == view.seat else NAMES[seat]
            assert line == f"Hearts were broken on trick {t} by {who}."
            seen_you |= who == "you"
            seen_other |= who != "you"
    assert seen_you and seen_other


def test_hearts_broken_negative():
    view, table = view_table(0, 0)
    assert _hearts_broken_sentence(view, NAMES) is None
    out = sentences(view, table, None, None, NAMES)
    assert not any(s.startswith("Hearts were broken") for s in out)


# F. points left

def test_points_left_count():
    for seed in range(20):
        for n in range(0, 52, 4):
            view, table = view_table(seed, n)
            played = {c for _, c in all_plays(view)}
            expect = (13 - sum(1 for c in played if c >= 39)) + \
                (0 if 36 in played else 13)
            line = _points_left_sentence(view)
            if expect == 0:
                assert line is None
            elif expect == 1:
                assert line == "1 point is still out of 26."
            else:
                assert line == f"{expect} points are still out of 26."


def test_points_left_zero_and_one():
    view, table = scan(lambda v, t: not (POINTS & ~played_mask(v)),
                       plays_range=range(30, 52, 1), seeds=range(200))
    assert _points_left_sentence(view) is None
    out = sentences(view, table, None, None, NAMES)
    assert not any("out of 26" in s for s in out)
    view, table = scan(lambda v, t: bin(POINTS & ~played_mask(v))
                       .count("1") == 1 and played_mask(v) & (1 << 36),
                       plays_range=range(30, 52, 1), seeds=range(300))
    assert _points_left_sentence(view) == "1 point is still out of 26."


POINTS = cards.POINTS_MASK


def played_mask(view):
    m = 0
    for _, c in all_plays(view):
        m |= 1 << c
    return m


# priority and trimming

def test_trim_drops_hearts_out_first_and_keeps_order():
    def cond(v, t):
        if not _hearts_sentence(v, t):
            return False
        return len(sentences(v, t, [0, 1], [1.0, 2.0], NAMES)) == \
            MAX_SENTENCES and not any(
                "still out;" in s
                for s in sentences(v, t, [0, 1], [1.0, 2.0], NAMES))
    view, table = scan(cond, plays_range=range(8, 48, 2))
    out = sentences(view, table, [0, 1], [1.0, 2.0], NAMES)
    assert len(out) == MAX_SENTENCES
    # comparison survives and stays last; Q♠ line survives
    assert out[-1].startswith("2♣ averages 1.0 points")
    from apps.table.explain import _queen_sentence
    q = _queen_sentence(view, table, NAMES)
    if q:
        assert q in out


def test_sparse_state_shows_every_applicable_line():
    # Early on the cap does not bind: every template that applies is shown.
    for seed in range(10):
        view, table = view_table(seed, 5)
        out = sentences(view, table, None, None, NAMES)
        if len(out) < MAX_SENTENCES:
            for line in (_your_voids_sentence(view), _follow_sentence(view),
                         _queen_danger_sentence(view, table),
                         _hearts_broken_sentence(view, NAMES),
                         _points_left_sentence(view)):
                if line:
                    assert line in out
