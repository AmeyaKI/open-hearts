from openhearts.engine import cards
from openhearts.engine.state import PlayerView


class RandomPlayer:
    """Uniform random over legal moves. Sanity floor for evaluation."""

    def __init__(self, rng):
        self.rng = rng

    def choose(self, view: PlayerView) -> int:
        options = cards.cards_in(view.legal_moves)
        return options[self.rng.integers(len(options))]
