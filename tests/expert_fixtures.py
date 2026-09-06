"""Reachable fixtures for the expert tests.

The rulebook (§4) demands that every final fixture be REACHABLE: a residual
position must come from a legal 52-card deal played legally to that point --
never a fabricated hand with the 2♣ still in it.  `reach()` builds such a
prefix by seeded depth-first search over legal plays and fails loudly if it
cannot.  Card notation: rank in 23456789TJQKA, suit in cdsh, e.g. "Qs 3d".
"""
import numpy as np

from openhearts.engine import cards
from openhearts.engine.game import legal_moves
from openhearts.engine.state import GameState

_RANKS = "23456789TJQKA"
_SUITS = "cdsh"
N, E, S, W = 0, 1, 2, 3


def card(s: str) -> int:
    r, su = s[0].upper(), s[1].lower()
    return _SUITS.index(su) * 13 + _RANKS.index(r)


def hand(s: str) -> int:
    m = 0
    for tok in s.split():
        m |= cards.bit(card(tok))
    return m


def name(c: int) -> str:
    return cards.card_name(c)


def full_state(hands):
    """A trick-0 state from four explicit 13-card hands (a full deal)."""
    assert len(hands) == 4
    total = 0
    for h in hands:
        assert bin(h).count("1") == 13, "each seat needs 13 cards"
        assert total & h == 0, "overlapping hands"
        total |= h
    assert total == cards.FULL_DECK
    st = GameState(hands=list(hands))
    for seat in range(4):
        if hands[seat] & cards.bit(cards.TWO_CLUBS):
            st.to_play = seat
    return st


class Unreachable(RuntimeError):
    pass


def reach(residual, leader, hearts_broken=None, seed=0,
          max_restarts=3000, node_cap=300):
    """Return a GameState at the start of a trick, `leader` to play, with
    exactly `residual` (4 bitmasks, equal sizes) left in the four hands.

    Every card outside the residual is dealt to a seat as a prefix card and
    played before the position; the queen is captured iff it is outside the
    residual; hearts are broken iff a heart is outside the residual (so
    `hearts_broken=False` is impossible when any heart is in the prefix, and
    is asserted).  Randomized restarts over the prefix assignment plus a SHALLOW DFS
    over play order (fast failure on a bad assignment beats deep search --
    measured: 300 nodes x 3000 restarts reaches every fixture below 1 s
    where 50,000 x 1,000 did not); raises `Unreachable` past the budget.
    """
    assert len(residual) == 4
    n = bin(residual[0]).count("1")
    assert all(bin(h).count("1") == n for h in residual), "unequal residual sizes"
    assert 1 <= n <= 12
    union = 0
    for h in residual:
        assert union & h == 0
        union |= h
    assert not union & cards.bit(cards.TWO_CLUBS), "2♣ cannot survive trick 1"
    prefix_cards = cards.cards_in(cards.FULL_DECK & ~union)
    hearts_in_prefix = any(cards.suit(c) == cards.HEARTS for c in prefix_cards)
    if hearts_broken is not None:
        assert hearts_broken == hearts_in_prefix, (
            "hearts_broken is fixed by which hearts sit in the prefix")
    k = 13 - n
    rng = np.random.default_rng(seed)

    for _ in range(max_restarts):
        order = list(rng.permutation(prefix_cards))
        pm = [0, 0, 0, 0]
        for i, c in enumerate(order):
            pm[i % 4] |= cards.bit(int(c))
        st = GameState(hands=[residual[s] | pm[s] for s in range(4)])
        for s in range(4):
            if pm[s] & cards.bit(cards.TWO_CLUBS):
                st.to_play = s
        nodes = [0]

        def dfs(state):
            if state.trick_number == k:
                return state if state.to_play == leader else None
            nodes[0] += 1
            if nodes[0] > node_cap:
                return None
            seat = state.to_play
            legal = legal_moves(state.hands[seat], tuple(state.current_trick),
                                state.hearts_broken, state.trick_number)
            legal &= pm[seat]
            opts = cards.cards_in(legal)
            rng.shuffle(opts)
            for c in opts:
                nxt = state.copy()
                nxt.play(int(c))
                got = dfs(nxt)
                if got is not None:
                    return got
            return None

        got = dfs(st)
        if got is not None:
            assert got.hands == list(residual) and not got.current_trick
            assert got.to_play == leader and got.trick_number == k
            assert got.hearts_broken == hearts_in_prefix
            return got
    raise Unreachable(f"no legal prefix found in {max_restarts} restarts")


def play(state, plays):
    """Play a scripted sequence of card names, asserting legality."""
    for s in plays.split():
        state.play(card(s))
    return state
