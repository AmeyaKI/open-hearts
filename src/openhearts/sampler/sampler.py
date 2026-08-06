"""Sample one concrete assignment of unseen cards to the three opponents.

Basic strategy: walk the unseen cards, assign each to an opponent with
probability proportional to the table, skipping opponents who are full or
void in that suit. Dead end => restart. Honesty note (also in README):
assigning card-by-card from the table's per-card probabilities is slightly
biased compared to sampling whole arrangements uniformly under the
constraints. Acceptable in practice; stated rather than hidden.
"""
import numpy as np

from openhearts.engine import cards


def sample_arrangement(table, rng, max_restarts=200):
    unseen = cards.cards_in(table.unseen_mask)
    for attempt in range(1, max_restarts + 1):
        remaining = list(table.hand_sizes)
        hands = [0, 0, 0]
        dead = False
        for c in unseen:
            w = table.probs[:, c].copy()
            for i in range(3):
                if remaining[i] == 0 or cards.suit(c) in table.voids[i]:
                    w[i] = 0.0
            total = w.sum()
            if total <= 0:
                dead = True
                break
            i = int(rng.choice(3, p=w / total))
            hands[i] |= cards.bit(c)
            remaining[i] -= 1
        if not dead and all(r == 0 for r in remaining):
            return hands, attempt
    return None
