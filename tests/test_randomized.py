import numpy as np
from openhearts.engine import cards
from openhearts.engine.game import deal, play_game
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.players.randomized import RandomizedHeuristic


def test_epsilon_zero_matches_heuristic_exactly():
    rng = np.random.default_rng(0)
    a = play_game(deal(np.random.default_rng(5)),
                  [RandomizedHeuristic(rng, epsilon=0.0) for _ in range(4)])
    b = play_game(deal(np.random.default_rng(5)),
                  [HeuristicPlayer() for _ in range(4)])
    assert a.history == b.history


def test_epsilon_one_always_deviates_when_possible():
    # with epsilon=1 the played card differs from the heuristic's choice
    # whenever >1 legal move exists; verify over a full game replay.
    rng = np.random.default_rng(1)
    state = deal(np.random.default_rng(6))
    h = HeuristicPlayer()
    p = RandomizedHeuristic(rng, epsilon=1.0)
    diffs = checks = 0
    while not state.is_over():
        seat = state.to_play
        view = state.view_for(seat)
        card = p.choose(view)
        if len(cards.cards_in(view.legal_moves)) > 1:
            checks += 1
            diffs += (card != h.choose(view))
        state.play(card)
    assert checks > 0 and diffs == checks


def test_games_stay_legal():
    rng = np.random.default_rng(2)
    players = [RandomizedHeuristic(rng, epsilon=0.2) for _ in range(4)]
    for seed in range(50):
        state = play_game(deal(np.random.default_rng(seed)), players)
        assert sum(state.scores) == 26
