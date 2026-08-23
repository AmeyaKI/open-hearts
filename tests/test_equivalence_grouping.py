"""Unit tests for equivalence-class card grouping (search/grouping.py).

Hand-built card examples. The interesting ones are the near-misses: a class
that LOOKS groupable by raw rank adjacency but is not, and one that looks
non-adjacent but is.
"""
import numpy as np
import pytest

from openhearts.engine import cards
from openhearts.engine.game import legal_moves
from openhearts.engine.kernel import jit_enabled
from openhearts.search import grouping

C, D, S, H = cards.CLUBS, cards.DIAMONDS, cards.SPADES, cards.HEARTS
R = {r: i for i, r in enumerate("23456789TJQKA")}


def card(rank_ch, suit) -> int:
    return suit * 13 + R[rank_ch]


def mask(*cs) -> int:
    m = 0
    for c in cs:
        m |= 1 << c
    return m


def classes(hand, dead=0, legal=None, current_trick=(), hearts_broken=True,
            trick_number=1):
    if legal is None:
        legal = legal_moves(hand, list(current_trick), hearts_broken,
                            trick_number)
    rep_mask, rep_of = grouping.equivalence_classes(hand, legal, dead)
    reps = cards.cards_in(rep_mask)
    return {r: grouping.class_members(rep_of, r) for r in reps}


def test_two_and_three_of_clubs_group():
    """2c/3c both strictly lowest: one class, represented by the 2c."""
    hand = mask(card("2", C), card("3", C), card("A", H))
    cls = classes(hand)
    assert cls[card("2", C)] == [card("2", C), card("3", C)]


def test_ten_and_jack_of_diamonds_group_with_nine_and_queen_played():
    """Td/Jd are already rank-adjacent -- the real test is the gap case."""
    hand = mask(card("T", D), card("J", D))
    dead = mask(card("9", D), card("Q", D))
    cls = classes(hand, dead=dead)
    assert cls[card("T", D)] == [card("T", D), card("J", D)]


def test_gap_spanned_by_a_dead_card():
    """Td and Qd group when the Jd is in a completed trick."""
    hand = mask(card("T", D), card("Q", D))
    cls = classes(hand, dead=mask(card("J", D)))
    assert cls[card("T", D)] == [card("T", D), card("Q", D)]


def test_gap_not_spanned_by_a_live_card():
    hand = mask(card("T", D), card("Q", D))
    cls = classes(hand)
    assert sorted(cls) == [card("T", D), card("Q", D)]


def test_card_in_the_current_trick_is_live_not_dead():
    """7d/9d are NOT interchangeable while the 8d sits on the table: the 7
    loses that trick and the 9 wins it. The same 8d in a COMPLETED trick is
    dead and they group."""
    hand = mask(card("7", D), card("9", D))
    trick = ((0, card("8", D)),)
    cls = classes(hand, dead=0, current_trick=trick, trick_number=1)
    assert sorted(cls) == [card("7", D), card("9", D)]

    cls2 = classes(hand, dead=mask(card("8", D)))
    assert cls2[card("7", D)] == [card("7", D), card("9", D)]


def test_queen_of_spades_is_never_grouped():
    hand = mask(card("J", S), card("Q", S), card("K", S))
    cls = classes(hand)
    assert sorted(cls) == [card("J", S), card("Q", S), card("K", S)]


def test_own_queen_of_spades_breaks_the_chain():
    """The counterexample: Js and Ks are rank-adjacent among LIVE cards once
    our own Qs is discounted -- but grouping them is unsound, because the
    order-preserving map would send the queen (13 points) onto the jack (0)."""
    hand = mask(card("J", S), card("Q", S), card("K", S), card("2", H))
    cls = classes(hand)
    assert card("J", S) in cls and cls[card("J", S)] == [card("J", S)]
    assert cls[card("K", S)] == [card("K", S)]


def test_own_queen_breaks_the_chain_on_trick_zero_where_it_is_illegal():
    """Chain over the HAND, not the legal mask: on the first trick the points
    filter hides the Qs, and chaining over legal cards would wrongly join
    Js to Ks."""
    hand = mask(card("2", C), card("J", S), card("Q", S), card("K", S))
    trick = ((1, card("2", C) + 1),)   # a club led, we are void... use spades
    # We are void in the led suit on trick 0, so we discard: the points filter
    # removes the Qs from the legal set but Js/Ks stay legal.
    hand2 = mask(card("J", S), card("Q", S), card("K", S))
    legal = legal_moves(hand2, [(1, card("5", C))], False, 0)
    assert (legal >> cards.QUEEN_SPADES) & 1 == 0, "Qs should be illegal here"
    cls = classes(hand2, legal=legal)
    assert sorted(cls) == [card("J", S), card("K", S)]
    assert cls[card("J", S)] == [card("J", S)]
    del trick


def test_king_and_ace_of_spades_group_when_the_queen_is_ours():
    hand = mask(card("Q", S), card("K", S), card("A", S))
    cls = classes(hand)
    # Qs breaks its own chain, but K-A are consecutive and neither is the Qs.
    assert cls[card("K", S)] == [card("K", S), card("A", S)]


def test_king_and_ace_of_spades_group_when_the_queen_is_played():
    hand = mask(card("K", S), card("A", S))
    cls = classes(hand, dead=mask(card("Q", S)))
    assert cls[card("K", S)] == [card("K", S), card("A", S)]


def test_jack_and_ace_group_when_queen_and_king_are_played():
    """The relaxed Ks/As rule: no special-casing needed."""
    hand = mask(card("J", S), card("A", S))
    cls = classes(hand, dead=mask(card("Q", S), card("K", S)))
    assert cls[card("J", S)] == [card("J", S), card("A", S)]


def test_hearts_group_with_each_other_but_never_across_suits():
    hand = mask(card("5", H), card("6", H), card("6", D), card("7", D))
    cls = classes(hand)
    assert cls[card("5", H)] == [card("5", H), card("6", H)]
    assert cls[card("6", D)] == [card("6", D), card("7", D)]
    assert len(cls) == 2


def test_representative_is_the_lowest_member():
    hand = mask(card("4", C), card("5", C), card("6", C))
    cls = classes(hand)
    assert list(cls) == [card("4", C)]


@pytest.mark.skipif(not jit_enabled(), reason="JIT disabled")
def test_python_and_njit_twins_agree():
    """Random hands/dead sets: the two implementations must agree exactly."""
    rng = np.random.default_rng(7)
    for _ in range(2000):
        perm = rng.permutation(52)
        n_hand = int(rng.integers(1, 14))
        n_dead = int(rng.integers(0, 30))
        hand = mask(*[int(c) for c in perm[:n_hand]])
        dead = mask(*[int(c) for c in perm[n_hand:n_hand + n_dead]])
        legal = hand
        py = grouping.equivalence_classes(hand, legal, dead)
        jt = grouping.equivalence_classes(hand, legal, dead, use_jit=True)
        assert py[0] == jt[0]
        assert np.array_equal(py[1], jt[1])
