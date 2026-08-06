from openhearts.engine import cards


def test_suit_and_rank_mapping():
    assert cards.suit(0) == 0 and cards.rank(0) == 0          # 2 of clubs
    assert cards.suit(36) == 2 and cards.rank(36) == 10       # queen of spades
    assert cards.suit(51) == 3 and cards.rank(51) == 12       # ace of hearts


def test_masks():
    assert cards.SUIT_MASK[0] == (1 << 13) - 1
    assert cards.HEARTS_MASK == ((1 << 13) - 1) << 39
    assert cards.POINTS_MASK == cards.HEARTS_MASK | (1 << 36)
    assert bin(cards.SUIT_MASK[1]).count("1") == 13


def test_bit_helpers():
    m = cards.bit(0) | cards.bit(36) | cards.bit(51)
    assert cards.cards_in(m) == [0, 36, 51]
    assert cards.bit(5) == 1 << 5


def test_card_name():
    assert cards.card_name(0) == "2c"
    assert cards.card_name(36) == "Qs"
    assert cards.card_name(51) == "Ah"
