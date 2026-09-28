"""B1 tuning (PHASE7_PLAN.md TASK B1): coordinate descent over OpenSpiel ISMCTS
knobs on TUNING deals 300000+ (a fresh stream never used for evaluation).

Objective: the ISMCTS seats' mean points per hand in the ours-minority protocol
against our shipped bot (lower = better for ISMCTS). Deliberately tuned AGAINST
US on held-out-from-evaluation deals: the evaluation rows (deals 100000+) never
see these deals, and the tuned config is frozen here before they run.

Order (frozen in the pre-registration): uct_c -> max_world_samples -> final
policy (all with the random-rollout evaluator, 1000 sims); then prior_eps for
the heur-prior rung (heuristic rollout, PUCT). Each config is one run_cbench
run with its own --partial-tag (resumable), so a pause loses nothing.

Usage (pyspiel venv):
  <venv>/bin/python experiments/tune_ismcts.py --stage knobs --deals 100 --workers 12
  <venv>/bin/python experiments/tune_ismcts.py --stage eps   --deals 100 --workers 12
Writes/appends results/b1_tuning.txt.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_cbench as cb  # noqa: E402

TUNE_SEED_BASE = 300_000
OUT = os.path.join(cb.RESULTS, "b1_tuning.txt")


def run_cfg(label, cfg, deals, workers, sims=1000):
    os.environ[cb._ISMCTS_ENV] = json.dumps(cfg)
    cb._PARTIAL_TAG = f"_tune_{label}"
    t0 = time.time()
    ours, ism, _, _ = cb.run(deals, workers, "ours-minority", 50, 20, sims,
                             TUNE_SEED_BASE, "honest")
    import numpy as np
    from openhearts.eval.stats import bootstrap_ci
    m, lo, hi = bootstrap_ci(np.asarray(ism))
    line = (f"{label:28s} ismcts {m:.3f} [{lo:.3f}, {hi:.3f}]  ours {np.mean(ours):.3f}"
            f"  n={len(ism)}  wall {time.time() - t0:.0f}s  cfg={json.dumps(cfg)}\n")
    with open(OUT, "a") as f:
        f.write(line)
    print(line, end="", flush=True)
    return m


def coordinate_descent(deals, workers):
    best = dict(cb.ISMCTS_DEFAULTS, rung="tuned")
    with open(OUT, "a") as f:
        f.write(f"# B1 tuning stage=knobs {time.strftime('%F %T')} deals={deals} seeds={TUNE_SEED_BASE}+ workers={workers}\n")
    for key, grid in (("uct_c", [0.5, 1.0, 2.0, 4.0]),
                      ("world_samples", [-1, 10, 50]),
                      ("final_policy", ["MAX_VISIT_COUNT", "MAX_VALUE"])):
        scores = {}
        for v in grid:
            cfg = dict(best, **{key: v})
            scores[v] = run_cfg(f"{key}={v}", cfg, deals, workers)
        best[key] = min(scores, key=scores.get)
        with open(OUT, "a") as f:
            f.write(f"# -> best {key} = {best[key]}\n")
    with open(OUT, "a") as f:
        f.write(f"# FROZEN tuned knobs: {json.dumps(best)}\n")
    return best


def tune_eps(deals, workers, best_knobs, sims):
    with open(OUT, "a") as f:
        f.write(f"# B1 tuning stage=eps {time.strftime('%F %T')} deals={deals} sims={sims}\n")
    scores = {}
    for eps in (0.1, 0.25, 0.5):
        cfg = dict(best_knobs, rung="heur-prior", child_selection="PUCT", prior_eps=eps)
        scores[eps] = run_cfg(f"prior_eps={eps}", cfg, deals, workers, sims=sims)
    best = min(scores, key=scores.get)
    with open(OUT, "a") as f:
        f.write(f"# FROZEN prior_eps = {best}\n")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["knobs", "eps"], required=True)
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--knobs-json", default=None, help="eps stage: the frozen tuned knobs")
    ap.add_argument("--sims", type=int, default=1000, help="eps stage: the equal-time simulation count")
    args = ap.parse_args()
    if args.stage == "knobs":
        coordinate_descent(args.deals, args.workers)
    else:
        assert args.knobs_json, "--knobs-json required for the eps stage"
        tune_eps(args.deals, args.workers, json.loads(args.knobs_json), args.sims)


if __name__ == "__main__":
    main()
