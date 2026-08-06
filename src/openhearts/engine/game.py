"""Dealing, legal moves, trick resolution, and the game loop."""
from . import cards
from .state import GameState


def legal_moves(hand: int, current_trick, hearts_broken: bool,
                trick_number: int) -> int:
    if current_trick:
        led = cards.suit(current_trick[0][1])
        follow = hand & cards.SUIT_MASK[led]
        if follow:
            if trick_number == 0:
                safe = follow & ~cards.POINTS_MASK
                return safe if safe else follow
            return follow
        if trick_number == 0:
            safe = hand & ~cards.POINTS_MASK
            return safe if safe else hand
        return hand
    # leading
    if trick_number == 0:
        return cards.bit(cards.TWO_CLUBS)
    if hearts_broken:
        return hand
    non_hearts = hand & ~cards.HEARTS_MASK
    return non_hearts if non_hearts else hand


def trick_winner(trick) -> int:
    led = cards.suit(trick[0][1])
    best_seat, best_rank = trick[0][0], cards.rank(trick[0][1])
    for seat, card in trick[1:]:
        if cards.suit(card) == led and cards.rank(card) > best_rank:
            best_seat, best_rank = seat, cards.rank(card)
    return best_seat


def trick_points(trick) -> int:
    pts = 0
    for _, card in trick:
        if cards.suit(card) == cards.HEARTS:
            pts += 1
        elif card == cards.QUEEN_SPADES:
            pts += 13
    return pts


def deal(rng) -> GameState:
    order = rng.permutation(52)
    hands = [0, 0, 0, 0]
    for i, card in enumerate(order):
        hands[i % 4] |= cards.bit(int(card))
    state = GameState(hands=hands)
    for seat in range(4):
        if hands[seat] & cards.bit(cards.TWO_CLUBS):
            state.to_play = seat
    return state


def play_game(state: GameState, players) -> GameState:
    while not state.is_over():
        seat = state.to_play
        card = players[seat].choose(state.view_for(seat))
        state.play(card)
    return state
