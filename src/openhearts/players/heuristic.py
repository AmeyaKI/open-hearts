"""Deterministic rule-based player.

Used both as the evaluation opponent and as the stand-in for ALL seats inside
imagined playouts. Deliberately has no randomness and never sees hidden cards.
"""
from openhearts.engine import cards
from openhearts.engine.state import PlayerView


class HeuristicPlayer:
    def choose(self, view: PlayerView) -> int:
        legal = cards.cards_in(view.legal_moves)
        if len(legal) == 1:
            return legal[0]
        if not view.current_trick:
            return self._lead(view, legal)
        led = cards.suit(view.current_trick[0][1])
        if cards.suit(legal[0]) == led:
            return self._follow(view, legal, led)
        return self._discard(view, legal)

    # -- leading -------------------------------------------------------
    def _lead(self, view, legal):
        seen = view.hand
        for _, c in view.history:
            seen |= cards.bit(c)
        by_suit = {}
        for c in legal:
            by_suit.setdefault(cards.suit(c), []).append(c)
        # avoid leading spades while holding Qs with K/A of spades still out
        if (cards.SPADES in by_suit and len(by_suit) > 1
                and view.hand & cards.bit(cards.QUEEN_SPADES)):
            high_spades_out = any(
                not (seen & cards.bit(c)) for c in (37, 38)  # Ks, As
            )
            if high_spades_out:
                del by_suit[cards.SPADES]
        best_suit = min(by_suit, key=lambda s: (len(by_suit[s]), s))
        return min(by_suit[best_suit])

    # -- following in the led suit ------------------------------------
    def _follow(self, view, legal, led):
        win_rank = max(
            cards.rank(c) for _, c in view.current_trick
            if cards.suit(c) == led
        )
        losers = [c for c in legal if cards.rank(c) < win_rank]
        winners = [c for c in legal if cards.rank(c) > win_rank]
        if not winners:  # can't win: shed the most dangerous loser
            if cards.QUEEN_SPADES in losers:
                return cards.QUEEN_SPADES
            return max(losers)
        if losers:       # duck as high as possible
            return max(losers)
        return min(winners)  # forced to win: minimize damage

    # -- discarding ----------------------------------------------------
    def _discard(self, view, legal):
        if cards.QUEEN_SPADES in legal:
            return cards.QUEEN_SPADES
        hearts = [c for c in legal if cards.suit(c) == cards.HEARTS]
        if hearts:
            return max(hearts)
        by_suit = {}
        for c in legal:
            by_suit.setdefault(cards.suit(c), []).append(c)
        longest = max(by_suit, key=lambda s: (len(by_suit[s]), -s))
        return max(by_suit[longest])
