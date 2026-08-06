from openhearts.engine import cards
from openhearts.engine.game import legal_moves
from openhearts.engine.cards import bit


def m(*cs):
    out = 0
    for c in cs:
        out |= bit(c)
    return out


def test_must_follow_led_suit():
    hand = m(5, 14, 40)                       # a club, a diamond, a heart
    trick = ((1, 3),)                         # club led
    assert legal_moves(hand, trick, False, 4) == m(5)


def test_discard_anything_when_void():
    hand = m(14, 40)                          # no clubs
    trick = ((1, 3),)                         # club led
    assert legal_moves(hand, trick, False, 4) == m(14, 40)


def test_first_trick_must_lead_two_of_clubs():
    hand = m(0, 5, 40)
    assert legal_moves(hand, (), False, 0) == m(0)


def test_no_heart_lead_before_broken():
    hand = m(5, 40, 45)
    assert legal_moves(hand, (), False, 4) == m(5)


def test_heart_lead_allowed_after_broken():
    hand = m(5, 40)
    assert legal_moves(hand, (), True, 4) == m(5, 40)


def test_heart_lead_allowed_when_only_hearts():
    hand = m(40, 45)
    assert legal_moves(hand, (), False, 4) == m(40, 45)


def test_queen_spades_lead_allowed_before_broken():
    hand = m(36, 40)                          # Qs is not a heart
    assert legal_moves(hand, (), False, 4) == m(36)


def test_no_points_discarded_on_first_trick():
    hand = m(14, 36, 40)                      # void in clubs
    trick = ((1, 0),)                         # 2c led
    assert legal_moves(hand, trick, False, 0) == m(14)


def test_points_on_first_trick_if_thats_all_you_hold():
    hand = m(36, 40, 45)
    trick = ((1, 0),)
    assert legal_moves(hand, trick, False, 0) == m(36, 40, 45)
