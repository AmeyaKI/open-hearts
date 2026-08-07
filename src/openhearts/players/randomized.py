"""Heuristic player with epsilon-probability random deviations.

Wraps a HeuristicPlayer: with probability 1-epsilon plays exactly what the
heuristic would; with probability epsilon plays a uniformly random OTHER
legal card. Sees only PlayerView.
"""
from openhearts.engine import cards
from openhearts.engine.state import PlayerView
from openhearts.players.heuristic import HeuristicPlayer


class RandomizedHeuristic:
    def __init__(self, rng, epsilon=0.1):
        self.rng = rng
        self.epsilon = epsilon
        self.heuristic = HeuristicPlayer()

    def choose(self, view: PlayerView) -> int:
        legal = cards.cards_in(view.legal_moves)
        if len(legal) == 1:
            return legal[0]
        choice = self.heuristic.choose(view)
        if self.rng.random() < self.epsilon:
            others = [c for c in legal if c != choice]
            return others[self.rng.integers(len(others))]
        return choice
