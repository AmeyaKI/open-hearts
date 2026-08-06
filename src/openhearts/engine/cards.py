"""Card constants and bit helpers. Cards are ints 0-51; hands are 52-bit ints."""

NUM_CARDS = 52
TWO_CLUBS = 0
QUEEN_SPADES = 36

SUIT_MASK = [((1 << 13) - 1) << (13 * s) for s in range(4)]
CLUBS, DIAMONDS, SPADES, HEARTS = range(4)
HEARTS_MASK = SUIT_MASK[HEARTS]
POINTS_MASK = HEARTS_MASK | (1 << QUEEN_SPADES)
FULL_DECK = (1 << 52) - 1

_RANKS = "23456789TJQKA"
_SUITS = "cdsh"


def suit(card: int) -> int:
    return card // 13


def rank(card: int) -> int:
    return card % 13


def bit(card: int) -> int:
    return 1 << card


def cards_in(mask: int) -> list[int]:
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def card_name(card: int) -> str:
    return _RANKS[rank(card)] + _SUITS[suit(card)]
