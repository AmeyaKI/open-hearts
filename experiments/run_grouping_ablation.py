"""Paired 100-deal mini-ablation: honest-FULL grouped vs ungrouped.

Both rows play the IDENTICAL deals (seeds 100000..100099, 4 rotations each)
against 3 heuristics, and each row's bot is seeded per (row-independent)
(deal, rotation) -- so the only difference between the rows is the
`group_equivalent` flag. Pre-registered stake: grouping is a theorem, so the
paired difference should be a small non-zero number driven only by the inner
seeds it no longer draws, and its 95% bootstrap CI must contain 0.

    .venv/bin/python experiments/run_grouping_ablation.py [--deals N]
                                                          [--workers K]
"""
import argparse
import concurrent.futures as cf
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

SEED0 = 100000


def _chunk(name, group, seeds):
    from openhearts.belief.table import Level
    from openhearts.eval.harness import rotated_match
    from openhearts.players.heuristic import HeuristicPlayer
    from openhearts.search.honest import HonestSearchPlayer

    it = iter([(s, r) for s in seeds for r in range(4)])

    def factory():
        s, r = next(it)
        # Row-independent seed: the two rows get the SAME bot rng.
        return HonestSearchPlayer(Level.FULL, 50, 20,
                                  np.random.default_rng([7, s, r]),
                                  fused=True, group_equivalent=group)

    return name, rotated_match(seeds, factory, lambda: HeuristicPlayer())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from openhearts.eval.stats import bootstrap_ci

    seeds = [SEED0 + i for i in range(args.deals)]
    chunks = [seeds[i:i + 5] for i in range(0, len(seeds), 5)]
    jobs = [(f"{'grouped' if g else 'ungrouped'}:{i}", g, c)
            for g in (False, True) for i, c in enumerate(chunks)]

    t0 = time.time()
    out = {"grouped": {}, "ungrouped": {}}
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_chunk, n, g, c) for n, g, c in jobs]
        for f in cf.as_completed(futs):
            name, per_deal = f.result()
            row, idx = name.split(":")
            out[row][int(idx)] = per_deal
    grouped = np.concatenate([out["grouped"][i] for i in range(len(chunks))])
    ungrouped = np.concatenate([out["ungrouped"][i]
                                for i in range(len(chunks))])
    diff = grouped - ungrouped

    for label, arr in (("ungrouped", ungrouped), ("grouped", grouped)):
        m, lo, hi = bootstrap_ci(arr)
        print(f"honest-FULL {label:9s}: {m:.3f} pts/hand  CI ({lo:.3f}, {hi:.3f})")
    m, lo, hi = bootstrap_ci(diff)
    print(f"paired diff (grouped - ungrouped): {m:+.4f}  "
          f"CI ({lo:+.4f}, {hi:+.4f})  -- contains 0: {lo <= 0 <= hi}")
    print(f"deals={len(seeds)}  wall={time.time() - t0:.0f}s  "
          f"workers={args.workers}")


if __name__ == "__main__":
    main()
