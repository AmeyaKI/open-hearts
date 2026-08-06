"""Headline result 2: does better guessing mean fewer points?

Runs the SearchPlayer at three belief levels (plus a heuristic row and a
random-legal floor) over an identical set of rotated deals, then sweeps the
sample count at the FULL level. Every configuration sees the same deals in the
same seats; confidence intervals resample deals, never individual games.

Determinism: each SearchPlayer/RandomPlayer gets an rng seeded from
(config id, deal seed, rotation), so a rerun reproduces every game exactly.
This relies on `rotated_match`'s documented iteration order (seeds outer,
rotations inner, bot_factory called once per game).

Usage:
    python experiments/run_ablation.py --probe   # timing probe only, then exit
    python experiments/run_ablation.py           # probe + full experiment
"""
import argparse
import itertools
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openhearts.belief.table import Level  # noqa: E402
from openhearts.eval.harness import rotated_match  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.players.random_player import RandomPlayer  # noqa: E402
from openhearts.search.decision import SearchPlayer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
N_DEALS = 500
DEAL_SEEDS = [100_000 + i for i in range(N_DEALS)]
MAIN_LEVELS = [("UNIFORM", Level.UNIFORM), ("VOIDS", Level.VOIDS),
               ("FULL", Level.FULL)]
MAIN_N_SAMPLES = 100
SWEEP_N = [10, 50, 100, 500]
PARTIAL = os.path.join(RESULTS, "ablation_partial.txt")


def _game_seed(config_id, deal_seed, rotation):
    """Deterministic per (config, deal seed, rotation)."""
    return np.random.default_rng([config_id, deal_seed, rotation])


def search_factory(config_id, level, n_samples, deal_seeds, tally=None):
    """Zero-arg factory producing a fresh, deterministically seeded bot.

    Each bot is discarded after its game, so its `fallbacks` / `failed_samples`
    counters would vanish with it. `tally` (a dict) collects the bots so the
    caller can sum them: UNIFORM ignores voids in the table while the sampler
    still masks them, so dead-end and fallback rates differ systematically
    across levels -- and a config that falls back to the heuristic often is
    partly measuring the heuristic, which must be visible in the report.
    """
    counter = itertools.count()
    def factory():
        i = next(counter)
        seed, rotation = deal_seeds[i // 4], i % 4
        bot = SearchPlayer(level, n_samples, _game_seed(config_id, seed, rotation))
        if tally is not None:
            tally.setdefault("bots", []).append(bot)
        return bot
    return factory


def random_factory(config_id, deal_seeds):
    counter = itertools.count()
    def factory():
        i = next(counter)
        seed, rotation = deal_seeds[i // 4], i % 4
        return RandomPlayer(_game_seed(config_id, seed, rotation))
    return factory


def _progress(label):
    def on_deal(done, total):
        if done % 25 == 0 or done == total:
            print(f"  [{label}] {done}/{total} deals", flush=True)
    return on_deal


def _append_partial(name, values):
    os.makedirs(RESULTS, exist_ok=True)
    with open(PARTIAL, "a") as f:
        f.write(name + " " + " ".join(f"{v:.6f}" for v in values) + "\n")
        f.flush()


def run_config(name, bot_factory, deal_seeds, tally=None):
    """Run one configuration; append per-deal values to the partial file."""
    print(f"[config] {name}: {len(deal_seeds)} deals x 4 rotations", flush=True)
    t0 = time.time()
    per_deal = rotated_match(deal_seeds, bot_factory,
                             lambda: HeuristicPlayer(), on_deal=_progress(name))
    elapsed = time.time() - t0
    mean, lo, hi = bootstrap_ci(per_deal)
    print(f"[config] {name}: mean={mean:.3f} CI=({lo:.3f},{hi:.3f}) "
          f"in {elapsed:.1f}s", flush=True)
    fallbacks = failed = None
    if tally is not None:
        bots = tally.get("bots", [])
        fallbacks = sum(b.fallbacks for b in bots)
        failed = sum(b.failed_samples for b in bots)
        print(f"[config] {name}: heuristic fallbacks={fallbacks}, "
              f"failed sample draws={failed}", flush=True)
    _append_partial(name, per_deal)
    return {"name": name, "per_deal": per_deal, "mean": mean, "lo": lo,
            "hi": hi, "seconds": elapsed, "fallbacks": fallbacks,
            "failed_samples": failed}


# ---------------------------------------------------------------- step 1
def timing_probe():
    """5 deals of FULL/n=100 vs heuristics; extrapolate the whole grid."""
    probe_seeds = [900_000 + i for i in range(5)]
    print("[probe] 5 deals x 4 rotations, SearchPlayer(FULL, n_samples=100)",
          flush=True)
    t0 = time.time()
    per_deal = rotated_match(
        probe_seeds, search_factory(999, Level.FULL, MAIN_N_SAMPLES, probe_seeds),
        lambda: HeuristicPlayer())
    elapsed = time.time() - t0
    n_games = 4 * len(probe_seeds)
    s_per_game = elapsed / n_games

    # Grid cost in "n=100 game equivalents". Playout cost is ~linear in
    # n_samples, so an n=k game costs about k/100 of an n=100 game.
    games = 4 * N_DEALS
    main_search = 3 * games                      # UNIFORM, VOIDS, FULL at n=100
    sweep = games * sum(n / MAIN_N_SAMPLES for n in SWEEP_N if n != MAIN_N_SAMPLES)
    equivalents = main_search + sweep            # heuristic/random rows are ~free
    est_seconds = equivalents * s_per_game

    print(f"[probe] mean bot points/deal = {per_deal.mean():.2f}", flush=True)
    print(f"[probe] {elapsed:.1f}s for {n_games} games "
          f"= {s_per_game:.2f}s/game ({elapsed/len(probe_seeds):.2f}s/deal)",
          flush=True)
    print(f"[probe] full grid ~= {equivalents:.0f} n=100-game equivalents",
          flush=True)
    print(f"[probe] estimated total runtime = {est_seconds/3600:.2f} hours",
          flush=True)
    print("[probe] note: per-level cost varies (UNIFORM ignores voids in the "
          "table while the sampler still masks them, so restart rates differ); "
          "treat the estimate as +/- a sizeable margin.", flush=True)
    return s_per_game, est_seconds


# ---------------------------------------------------------------- steps 2-4
def main_ablation():
    rows = []
    for cid, (name, level) in enumerate(MAIN_LEVELS):
        tally = {}
        rows.append(run_config(
            f"search-{name}-n{MAIN_N_SAMPLES}",
            search_factory(cid, level, MAIN_N_SAMPLES, DEAL_SEEDS, tally),
            DEAL_SEEDS, tally))
    rows.append(run_config("heuristic", lambda: HeuristicPlayer(), DEAL_SEEDS))
    rows.append(run_config("random-legal", random_factory(90, DEAL_SEEDS),
                           DEAL_SEEDS))
    return rows


def sweep(main_rows):
    """FULL level at n_samples in SWEEP_N; reuse the main n=100 FULL row."""
    reuse = next(r for r in main_rows if r["name"] == f"search-FULL-n{MAIN_N_SAMPLES}")
    rows = []
    for cid, n in enumerate(SWEEP_N, start=10):
        if n == MAIN_N_SAMPLES:
            rows.append({**reuse, "n_samples": n, "reused": True})
            continue
        tally = {}
        r = run_config(f"sweep-FULL-n{n}",
                       search_factory(cid, Level.FULL, n, DEAL_SEEDS, tally),
                       DEAL_SEEDS, tally)
        rows.append({**r, "n_samples": n, "reused": False})
    return rows


def write_outputs(main_rows, sweep_rows, probe):
    os.makedirs(RESULTS, exist_ok=True)
    s_per_game, est_seconds = probe
    lines = [
        "open-hearts ablation (headline result 2)",
        "Lower points per hand is better. 26 points are dealt out per hand,",
        "so 6.5 is the symmetric break-even; a bot below 6.5 is beating the",
        "three heuristic opponents it plays against.",
        "",
        f"deals: {N_DEALS} (seeds {DEAL_SEEDS[0]}..{DEAL_SEEDS[-1]}), "
        f"each played 4x with the bot rotated through every seat",
        "identical deals for every configuration; CIs are 10,000 bootstrap",
        "resamples over DEALS (not games): the 4 rotations of one deal share",
        "that deal's luck and count as one observation.",
        f"timing probe: {s_per_game:.2f}s per n=100 game, "
        f"estimated grid {est_seconds/3600:.2f}h",
        "",
        "== main ablation (SearchPlayer n_samples=100 vs 3 heuristics) ==",
        "'fallbk' counts decisions where the sampler failed often enough that",
        "the bot fell back to the plain heuristic; a large count means that row",
        "is partly measuring the heuristic, not the search.",
        f"{'config':<24}{'mean':>8}{'lo95':>9}{'hi95':>9}{'secs':>10}"
        f"{'fallbk':>9}{'failsmp':>9}",
    ]
    for r in main_rows:
        fb = "-" if r.get("fallbacks") is None else str(r["fallbacks"])
        fs = "-" if r.get("failed_samples") is None else str(r["failed_samples"])
        lines.append(f"{r['name']:<24}{r['mean']:>8.3f}{r['lo']:>9.3f}"
                     f"{r['hi']:>9.3f}{r['seconds']:>10.1f}{fb:>9}{fs:>9}")
    lines += ["",
              "== sample-count sweep (FULL level, same deals) ==",
              "the n=100 row is the main ablation's FULL row reused verbatim "
              "(same deals, same seeds, same games).",
              f"{'n_samples':<12}{'mean':>8}{'lo95':>9}{'hi95':>9}{'reused':>9}"]
    for r in sweep_rows:
        lines.append(f"{r['n_samples']:<12}{r['mean']:>8.3f}{r['lo']:>9.3f}"
                     f"{r['hi']:>9.3f}{str(r['reused']):>9}")
    with open(os.path.join(RESULTS, "ablation.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # bar chart with CI error bars
    names = [r["name"] for r in main_rows]
    means = [r["mean"] for r in main_rows]
    err = np.array([[r["mean"] - r["lo"] for r in main_rows],
                    [r["hi"] - r["mean"] for r in main_rows]])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, means, yerr=err, capsize=5, color="#4C72B0")
    ax.axhline(6.5, color="grey", linestyle="--", label="6.5 = symmetric break-even")
    ax.set_ylabel("mean points per hand (lower is better)")
    ax.set_title(f"Belief-level ablation, {N_DEALS} deals x 4 rotations")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "ablation.png"), dpi=150)
    plt.close(fig)

    # sweep line chart
    xs = [r["n_samples"] for r in sweep_rows]
    ys = [r["mean"] for r in sweep_rows]
    serr = np.array([[r["mean"] - r["lo"] for r in sweep_rows],
                     [r["hi"] - r["mean"] for r in sweep_rows]])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, ys, yerr=serr, marker="o", capsize=4, color="#55A868")
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("n_samples (log scale)")
    ax.set_ylabel("mean points per hand (lower is better)")
    ax.set_title("FULL level: returns vs sample count")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "sweep.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="run only the timing probe (5 deals) and exit")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    probe = timing_probe()
    if args.probe:
        print("[probe] --probe given: stopping before the full experiment.",
              flush=True)
        return
    if probe[1] > 12 * 3600:
        print(f"\nSTOP: estimated runtime {probe[1]/3600:.2f}h exceeds the "
              "~12 hour budget. Per the plan, the project owner chooses the "
              "run sizes -- not this script. Nothing has been shrunk.",
              flush=True)
        return
    with open(PARTIAL, "w") as f:   # fresh per run; it is a crash-diagnosis log
        f.write(f"# run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    main_rows = main_ablation()
    sweep_rows = sweep(main_rows)
    write_outputs(main_rows, sweep_rows, probe)


if __name__ == "__main__":
    main()
