import numpy as np
from openhearts.engine import cards
from openhearts.engine.cards import bit
from openhearts.engine.state import PlayerView
from openhearts.engine.game import deal, play_game, legal_moves
from openhearts.players.heuristic import HeuristicPlayer


def make_view(hand, trick=(), hearts_broken=True, trick_number=5):
    return PlayerView(
        seat=0, hand=hand, history=(), current_trick=tuple(trick),
        hearts_broken=hearts_broken, trick_number=trick_number,
        scores=(0, 0, 0, 0),
        legal_moves=legal_moves(hand, tuple(trick), hearts_broken, trick_number),
    )


def test_discard_queen_of_spades_first():
    hand = bit(36) | bit(50) | bit(5)
    v = make_view(hand, trick=((1, 14),))     # diamond led, we are void
    assert HeuristicPlayer().choose(v) == 36


def test_discard_high_heart_when_no_queen():
    hand = bit(50) | bit(40) | bit(5)
    v = make_view(hand, trick=((1, 14),))
    assert HeuristicPlayer().choose(v) == 50


def test_duck_under_current_winner():
    # club led with the 9 (card 7); we hold 3c(1), 8c(6), Kc(11)
    hand = bit(1) | bit(6) | bit(11)
    v = make_view(hand, trick=((1, 7),))
    assert HeuristicPlayer().choose(v) == 6   # highest club that still loses


def test_forced_win_takes_lowest_winner():
    hand = bit(9) | bit(11)                   # Jc, Kc
    v = make_view(hand, trick=((1, 7),))      # 9c winning
    assert HeuristicPlayer().choose(v) == 9


def test_lead_lowest_in_shortest_suit():
    hand = bit(14) | bit(3) | bit(4) | bit(5)  # one diamond, three clubs
    v = make_view(hand, trick=())
    assert HeuristicPlayer().choose(v) == 14


def test_mirror_match_averages_about_6_5():
    # 4 identical heuristics: by symmetry each should score ~6.5/hand.
    # If not, the engine or heuristic is broken. Deterministic given the seed.
    rng = np.random.default_rng(7)
    players = [HeuristicPlayer() for _ in range(4)]
    totals = np.zeros(4)
    n = 400
    for _ in range(n):
        state = play_game(deal(rng), players)
        totals += state.scores
    means = totals / n
    assert abs(means.mean() - 6.5) < 1e-9      # always true: 26/4
    assert means.max() - means.min() < 1.5     # no seat systematically dominates
