"""Belief table: P(opponent i holds card c), from one observer's view only.

Built exclusively from PlayerView so inference cannot see hidden hands.
Zeros are permanent: seen cards and void suits stay zero through all
rebalancing. A wrongly-zeroed entry is unrecoverable, so we assert against it.

Known honesty note (also in README): FULL rebalancing finds the table
nearest the starting point that satisfies both constraint sets (this is
iterative proportional fitting). It is exact for the constraints we feed it;
it does not model HOW opponents choose plays, only what they can hold.
"""
from enum import Enum

import numpy as np

from openhearts.engine import cards
from openhearts.engine.state import PlayerView


class Level(Enum):
    UNIFORM = "uniform"
    VOIDS = "voids"
    FULL = "full"


class BeliefTable:
    def __init__(self, probs, voids, hand_sizes, opponent_seats, unseen_mask):
        self.probs = probs
        self.voids = voids
        self.hand_sizes = hand_sizes
        self.opponent_seats = opponent_seats
        self.unseen_mask = unseen_mask

    @classmethod
    def from_view(cls, view: PlayerView, level: Level) -> "BeliefTable":
        opponent_seats = [(view.seat + 1 + i) % 4 for i in range(3)]

        seen = view.hand
        all_plays = list(view.history) + list(view.current_trick)
        for _, c in all_plays:
            seen |= cards.bit(c)
        unseen_mask = cards.FULL_DECK & ~seen

        # how many cards each opponent still holds
        played_count = {s: 0 for s in range(4)}
        for s, _ in all_plays:
            played_count[s] += 1
        hand_sizes = [13 - played_count[s] for s in opponent_seats]

        # voids: failed to follow the led suit => none of that suit, ever
        voids = [set(), set(), set()]
        trick = []
        for s, c in all_plays:
            if trick and cards.suit(c) != cards.suit(trick[0][1]):
                if s in opponent_seats:
                    voids[opponent_seats.index(s)].add(
                        cards.suit(trick[0][1])
                    )
            trick.append((s, c))
            if len(trick) == 4:
                trick = []
        # NOTE: view.current_trick is included in the scan; a partial trick
        # still reveals voids for those who already played to it.

        probs = np.zeros((3, 52))
        for c in cards.cards_in(unseen_mask):
            probs[:, c] = 1.0

        if level in (Level.VOIDS, Level.FULL):
            for i in range(3):
                for s in voids[i]:
                    probs[i, 13 * s: 13 * s + 13] = 0.0

        # guard: every unseen card must have at least one possible holder
        col = probs.sum(axis=0)
        for c in cards.cards_in(unseen_mask):
            assert col[c] > 0, f"card {c} has no possible holder (bad zeroing)"

        if level == Level.FULL:
            probs = _rebalance(probs, hand_sizes, unseen_mask)
        else:
            probs /= np.where(col > 0, col, 1.0)  # columns sum to 1

        return cls(probs, voids, hand_sizes, opponent_seats, unseen_mask)


def _rebalance(probs, hand_sizes, unseen_mask,
               tol=1e-9, coarse_tol=1e-4, coarse_after=2000, max_iters=100000):
    """Alternately scale rows to hand sizes and unseen columns to 1.

    Honesty note on the two tolerances. Late in a hand the constraints can
    combinatorially FORCE an entry to zero (e.g. three cards left, and only
    one opponent is not void in spades, so the two spades must go to the
    other two). The limit of the iteration then sits on the boundary of the
    simplex, where iterative proportional fitting degrades from geometric to
    sublinear convergence: deviation falls off like ~1/k, so reaching 1e-9
    would take order 1e9 passes. Measured on real views this affects roughly
    8% of completed-trick boundaries, concentrated in tricks 8-13.

    So: we stop at `tol` when it is reachable (the overwhelmingly common
    case, a few dozen passes), and after `coarse_after` passes accept
    `coarse_tol` instead. `coarse_tol` is two orders of magnitude tighter
    than any tolerance the tests or metrics rely on. Non-convergence past
    that is still a hard error - the guard is loosened, not removed. Fixing
    the underlying case properly means exact enumeration of the forced
    assignments, which is deliberately out of scope for this phase.
    """
    sizes = np.array(hand_sizes, dtype=float)
    unseen_cols = np.array(cards.cards_in(unseen_mask))
    if unseen_cols.size == 0:
        return probs
    for k in range(max_iters):
        row = probs.sum(axis=1)
        scale = np.where(row > 0, sizes / np.where(row > 0, row, 1.0), 1.0)
        probs = probs * scale[:, None]
        col = probs[:, unseen_cols].sum(axis=0)
        assert (col > 0).all(), "column collapsed to zero during rebalance"
        probs[:, unseen_cols] /= col
        dev = max(
            np.abs(probs.sum(axis=1) - sizes).max(),
            np.abs(probs[:, unseen_cols].sum(axis=0) - 1.0).max(),
        )
        if dev < tol or (k + 1 >= coarse_after and dev < coarse_tol):
            break
    else:
        raise AssertionError(f"rebalance did not converge (dev={dev})")
    assert (probs >= 0).all() and (probs <= 1 + 1e-9).all()
    return probs
