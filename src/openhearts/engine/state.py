"""Full game state (private to engine/eval) and the filtered view players receive.

Hard rule: player and belief code must only ever touch PlayerView. The ONLY
constructor of PlayerView is GameState.view_for, so all filtering lives here.
"""
from dataclasses import dataclass, field

from . import cards


@dataclass(frozen=True)
class PlayerView:
    seat: int
    hand: int
    history: tuple  # ((seat, card), ...) completed tricks, play order
    current_trick: tuple  # ((seat, card), ...) so far this trick
    hearts_broken: bool
    trick_number: int
    scores: tuple
    legal_moves: int


@dataclass
class GameState:
    hands: list  # 4 bitmasks
    history: list = field(default_factory=list)
    current_trick: list = field(default_factory=list)
    hearts_broken: bool = False
    trick_number: int = 0
    scores: list = field(default_factory=lambda: [0, 0, 0, 0])
    to_play: int = 0

    def view_for(self, seat: int) -> PlayerView:
        from .game import legal_moves  # avoid import cycle
        return PlayerView(
            seat=seat,
            hand=self.hands[seat],
            history=tuple(self.history),
            current_trick=tuple(self.current_trick),
            hearts_broken=self.hearts_broken,
            trick_number=self.trick_number,
            scores=tuple(self.scores),
            legal_moves=legal_moves(
                self.hands[seat], tuple(self.current_trick),
                self.hearts_broken, self.trick_number,
            ),
        )

    def is_over(self) -> bool:
        return all(h == 0 for h in self.hands) and not self.current_trick

    def copy(self) -> "GameState":
        return GameState(
            hands=list(self.hands),
            history=list(self.history),
            current_trick=list(self.current_trick),
            hearts_broken=self.hearts_broken,
            trick_number=self.trick_number,
            scores=list(self.scores),
            to_play=self.to_play,
        )

    def play(self, card: int) -> None:
        from .game import legal_moves, trick_winner, trick_points
        seat = self.to_play
        legal = legal_moves(self.hands[seat], tuple(self.current_trick),
                            self.hearts_broken, self.trick_number)
        assert legal & cards.bit(card), (
            f"illegal card {card} by seat {seat}"
        )
        self.hands[seat] &= ~cards.bit(card)
        self.current_trick.append((seat, card))
        if cards.suit(card) == cards.HEARTS:
            self.hearts_broken = True
        if len(self.current_trick) == 4:
            winner = trick_winner(self.current_trick)
            self.scores[winner] += trick_points(self.current_trick)
            self.history.extend(self.current_trick)
            self.current_trick = []
            self.trick_number += 1
            self.to_play = winner
        else:
            self.to_play = (seat + 1) % 4
