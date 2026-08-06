import pytest
import numpy as np
from openhearts.engine import cards
from openhearts.engine.game import deal, play_game, legal_moves
from openhearts.players.random_player import RandomPlayer


def _run_games(n_games: int, seed: int):
    rng = np.random.default_rng(seed)
    players = [RandomPlayer(rng) for _ in range(4)]
    for _ in range(n_games):
        state = play_game(deal(rng), players)
        played = [card for _, card in state.history]
        # every card played exactly once
        assert sorted(played) == list(range(52))
        # each player played exactly 13 cards
        counts = [0, 0, 0, 0]
        for seat, _ in state.history:
            counts[seat] += 1
        assert counts == [13, 13, 13, 13]
        # 2 of clubs opened the game
        assert state.history[0][1] == cards.TWO_CLUBS
        # points always total exactly 26
        assert sum(state.scores) == 26
        # (illegal moves impossible: GameState.play asserts legality on every card)


def test_invariants_1k_games():
    _run_games(1_000, seed=12345)


@pytest.mark.slow
def test_invariants_100k_games():
    # Run before each results milestone: python -m pytest -m slow
    _run_games(100_000, seed=54321)
