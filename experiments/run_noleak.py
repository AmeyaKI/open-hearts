"""Corrected ablation control: UNIFORM beliefs with a truly uninformed sampler.

The main ablation's sampler refuses to deal cards into observed void suits at
EVERY belief level, leaking the strongest evidence source into the "UNIFORM"
row. This runs the same 500 deals x 4 rotations with that leak closed
(sampler_respects_voids=False), giving the honest no-inference control.
"""
import concurrent.futures as cf
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import run_ablation as ra  # noqa: E402

CHUNK = 25
WORKERS = 12
CONFIG_ID = 70  # distinct from every main-ablation config id


def worker(start, chunk_seeds):
    import itertools
    from openhearts.belief.table import Level
    from openhearts.eval.harness import rotated_match
    from openhearts.players.heuristic import HeuristicPlayer
    from openhearts.search.decision import SearchPlayer

    counter = itertools.count()
    bots = []

    def factory():
        i = next(counter)
        seed, rotation = chunk_seeds[i // 4], i % 4
        bot = SearchPlayer(Level.UNIFORM, ra.MAIN_N_SAMPLES,
                           ra._game_seed(CONFIG_ID, seed, rotation),
                           sampler_respects_voids=False)
        bots.append(bot)
        return bot

    per_deal = rotated_match(chunk_seeds, factory, lambda: HeuristicPlayer())
    return (start, per_deal, sum(b.fallbacks for b in bots),
            sum(b.failed_samples for b in bots))


def main():
    from openhearts.eval.stats import bootstrap_ci
    per_deal = np.zeros(len(ra.DEAL_SEEDS))
    fallbacks = failed = 0
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        jobs = [pool.submit(worker, start, chunk)
                for start, chunk in ((i, ra.DEAL_SEEDS[i:i + CHUNK])
                                     for i in range(0, len(ra.DEAL_SEEDS), CHUNK))]
        for done, fut in enumerate(cf.as_completed(jobs), 1):
            start, vals, fb, fs = fut.result()
            per_deal[start:start + len(vals)] = vals
            fallbacks += fb
            failed += fs
            print(f"[{done}/{len(jobs)}] deals {start}-{start + len(vals) - 1} "
                  f"done | {time.time() - t0:.0f}s", flush=True)
    mean, lo, hi = bootstrap_ci(per_deal)
    lines = [
        "corrected control: search-UNIFORM-noleak-n100",
        f"same {len(ra.DEAL_SEEDS)} deals (seeds {ra.DEAL_SEEDS[0]}.."
        f"{ra.DEAL_SEEDS[-1]}), 4 rotations, vs 3 heuristics",
        "sampler_respects_voids=False: imagined worlds may contradict observed",
        "voids, so NO void evidence reaches this configuration at all.",
        f"mean={mean:.3f} lo95={lo:.3f} hi95={hi:.3f} "
        f"fallbacks={fallbacks} failed_samples={failed}",
    ]
    out = os.path.join(ra.RESULTS, "ablation_noleak.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
