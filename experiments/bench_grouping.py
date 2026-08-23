"""Timing + candidate-count report for equivalence-class card grouping.

honest-FULL 50x20, fused, 1 seat (the other three are the heuristic), with and
without `group_equivalent`. Reports s/game, ms/decision, and mean candidates
per decision overall and by trick bucket.

    .venv/bin/python experiments/bench_grouping.py [n_games]
"""
import sys
import time

import numpy as np

from openhearts.belief.table import Level
from openhearts.engine import cards
from openhearts.engine.game import deal
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.search.honest import HonestSearchPlayer


def run(n_games, group, seed0=50000):
    t = 0.0
    n_dec = 0
    n_cand = 0
    buckets = {}
    for g in range(n_games):
        rng = np.random.default_rng(1000 + g)
        me = HonestSearchPlayer(Level.FULL, 50, 20, rng, fused=True,
                                group_equivalent=group)
        players = [me] + [HeuristicPlayer() for _ in range(3)]
        state = deal(np.random.default_rng(seed0 + g))
        while not state.is_over():
            seat = state.to_play
            view = state.view_for(seat)
            if seat == 0 and len(cards.cards_in(view.legal_moves)) > 1:
                t0 = time.perf_counter()
                card = me.choose(view)
                t += time.perf_counter() - t0
                n_dec += 1
                k = len(me.last_candidates)
                n_cand += k
                b = buckets.setdefault(view.trick_number // 4, [0, 0])
                b[0] += k
                b[1] += 1
            else:
                card = players[seat].choose(view)
            state.play(card)
    return t, n_dec, n_cand, buckets


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run(1, True)          # warm the JIT
    rows = []
    for group in (False, True):
        t, n_dec, n_cand, buckets = run(n_games, group)
        rows.append((group, t, n_dec, n_cand, buckets))
        print(f"group_equivalent={group}: {t / n_games:.3f} s/game, "
              f"{1000 * t / n_dec:.1f} ms/decision, "
              f"{n_cand / n_dec:.2f} candidates/decision ({n_dec} decisions)")
        for k in sorted(buckets):
            b = buckets[k]
            print(f"    tricks {4 * k}-{4 * k + 3}: {b[0] / b[1]:.2f} "
                  f"candidates ({b[1]} decisions)")
    a, b = rows[0][1], rows[1][1]
    print(f"speedup: {a / b:.3f}x")


if __name__ == "__main__":
    main()
