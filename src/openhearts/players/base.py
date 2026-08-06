from typing import Protocol
from openhearts.engine.state import PlayerView


class Player(Protocol):
    def choose(self, view: PlayerView) -> int: ...
