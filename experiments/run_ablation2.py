"""Phase 2 ablation (headline result: do beliefs pay now?).

Runs `HonestSearchPlayer` (src/openhearts/search/honest.py) at three belief
levels vs 3 plain HeuristicPlayer opponents, on the same 500 deal seeds used
by Phase 1's ablation (run_ablation.DEAL_SEEDS), plus a `phase1-FULL-bridge`
row that reruns the ORIGINAL Phase-1 `SearchPlayer` at FULL/n_samples=100
with the exact seeding run_ablation.py used for its FULL row (config_id=2,
same deal seeds, same per-game rng formula) -- if the harness and seeding are
unchanged this row must reproduce ablation.txt's search-FULL-n100 mean
(3.267) bitwise, and is the sanity anchor for everything else in this file.

Rows:
    1. honest-UNIFORM-noleak  (HonestSearchPlayer, Level.UNIFORM,
                                sampler_respects_voids=False)
    2. honest-VOIDS           (Level.VOIDS)
    3. honest-FULL            (Level.FULL)
    4. phase1-FULL-bridge     (Phase-1 SearchPlayer, Level.FULL, n_samples=100,
                                config_id=2 -- bitwise bridge to ablation.txt)

Honest rows use n_outer=50, n_inner=20 (the parameters Task 3's commit timed),
fresh config_ids 300/301/302, and per-game rng seeded via
run_ablation._game_seed(config_id, deal_seed, rotation) -- identical
seeding discipline to Phase 1's ablation.

Determinism / opponent contract: identical to run_ablation.py -- opponents
are plain `HeuristicPlayer()` (free to construct, no seed needed),
`bot_factory()` is called once per game in `rotated_match`'s documented
iteration order (seeds outer, rotations inner).

Budget rule (PHASE23_PLAN Global Constraints): timing probe first (5 deals,
honest-FULL). Target 500 deals; if the projected total exceeds 2h, drop to
250 deals and say so prominently in the output file; if 250 is still
projected over 2h, print a STOP message and exit WITHOUT running (the lead
consults the owner) -- the bridge row is cheap and always runs at whatever
deal count is chosen for the honest rows.

Usage:
    python experiments/run_ablation2.py --smoke   # 2 deals/row, serial, exit
    python experiments/run_ablation2.py           # probe + full run (LEAD ONLY)
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_ablation as ra  # noqa: E402

from openhearts.belief.table import Level  # noqa: E402
from openhearts.eval.harness import rotated_match  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.search.decision import SearchPlayer  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
PARTIAL = os.path.join(RESULTS, "ablation2_partial.txt")

N_OUTER = 50
N_INNER = 20
MAIN_N_SAMPLES = 100          # bridge row's n_samples, matching Phase 1
N_DEALS_TARGET = 500
BUDGET_SECONDS = 2 * 3600
WORKERS = 12
CHUNK = 25
MEM_LIMIT_GB = 100.0

# Fresh config_ids (300..302); the bridge row reuses config_id=2, the exact
# id run_ablation.search_factory used for its FULL row (MAIN_LEVELS index 2).
HONEST_LEVELS = [("honest-UNIFORM-noleak", Level.UNIFORM, False, 300),
                 ("honest-VOIDS", Level.VOIDS, True, 301),
                 ("honest-FULL", Level.FULL, True, 302)]
BRIDGE_CONFIG_ID = 2

# Phase-1 reference numbers, hardcoded from results/ablation.txt and
# results/ablation_noleak.txt, for the ghosted comparison bars.
PHASE1_REFERENCE = [
    ("search-UNIFORM-noleak", 3.371),
    ("search-VOIDS", 3.139),
    ("search-FULL", 3.267),
]


def honest_factory(config_id, level, sampler_respects_voids, deal_seeds,
                   tally=None):
    """1 call/game (harness contract), seeded like run_ablation.search_factory."""
    counter = itertools.count()
    def factory():
        i = next(counter)
        seed, rotation = deal_seeds[i // 4], i % 4
        bot = HonestSearchPlayer(level, N_OUTER, N_INNER,
                                 ra._game_seed(config_id, seed, rotation),
                                 sampler_respects_voids)
        if tally is not None:
            tally.setdefault("bots", []).append(bot)
        return bot
    return factory


def bridge_factory(config_id, deal_seeds, tally=None):
    """Bitwise-identical to run_ablation.search_factory(2, Level.FULL, 100, ...)."""
    counter = itertools.count()
    def factory():
        i = next(counter)
        seed, rotation = deal_seeds[i // 4], i % 4
        bot = SearchPlayer(Level.FULL, MAIN_N_SAMPLES,
                           ra._game_seed(config_id, seed, rotation))
        if tally is not None:
            tally.setdefault("bots", []).append(bot)
        return bot
    return factory


CONFIGS = (
    [(name, "honest", cid, level.name, respects)
     for name, level, respects, cid in HONEST_LEVELS]
    + [("phase1-FULL-bridge", "bridge", BRIDGE_CONFIG_ID, "FULL", True)]
)


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


def _make_bot_factory(kind, config_id, level_name, respects, deal_seeds, tally):
    if kind == "honest":
        return honest_factory(config_id, Level[level_name], respects,
                              deal_seeds, tally)
    if kind == "bridge":
        return bridge_factory(config_id, deal_seeds, tally)
    raise ValueError(kind)


def _counters(kind, tally):
    """Sum (fallbacks, failed_samples, inner_fallbacks, inner_failed_samples)."""
    bots = tally.get("bots", []) if tally is not None else []
    fb = sum(b.fallbacks for b in bots)
    fs = sum(b.failed_samples for b in bots)
    if kind == "honest":
        ifb = sum(b.inner_fallbacks for b in bots)
        ifs = sum(b.inner_failed_samples for b in bots)
    else:
        ifb = ifs = None
    return fb, fs, ifb, ifs


def run_config_serial(name, kind, config_id, level_name, respects, deal_seeds):
    """Serial (used by --smoke); mirrors run_bprobe.run_config_serial."""
    tally = {}
    bot_factory = _make_bot_factory(kind, config_id, level_name, respects,
                                    deal_seeds, tally)
    print(f"[config] {name}: {len(deal_seeds)} deals x 4 rotations", flush=True)
    t0 = time.time()
    per_deal = rotated_match(deal_seeds, bot_factory,
                             lambda: HeuristicPlayer(), on_deal=_progress(name))
    elapsed = time.time() - t0
    mean, lo, hi = bootstrap_ci(per_deal)
    fb, fs, ifb, ifs = _counters(kind, tally)
    print(f"[config] {name}: mean={mean:.3f} CI=({lo:.3f},{hi:.3f}) "
          f"in {elapsed:.1f}s fallbacks={fb} failed={fs} "
          f"inner_fallbacks={ifb} inner_failed={ifs}", flush=True)
    _append_partial(name, per_deal)
    return {"name": name, "per_deal": per_deal, "mean": mean, "lo": lo,
            "hi": hi, "seconds": elapsed, "fallbacks": fb, "failed_samples": fs,
            "inner_fallbacks": ifb, "inner_failed_samples": ifs}


# ---------------------------------------------------------------- parallel
def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i:i + size]


def worker(name, kind, config_id, level_name, respects, start, chunk_seeds):
    """Run one chunk of deals for one configuration in a worker process."""
    tally = {}
    bot_factory = _make_bot_factory(kind, config_id, level_name, respects,
                                    chunk_seeds, tally)
    per_deal = rotated_match(chunk_seeds, bot_factory, lambda: HeuristicPlayer())
    fb, fs, ifb, ifs = _counters(kind, tally)
    return (name, start, per_deal, fb, fs, ifb, ifs)


def total_rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


def run_all(deal_seeds, workers):
    import concurrent.futures as cf
    results = {name: np.zeros(len(deal_seeds)) for name, *_ in CONFIGS}
    tallies = {name: [0, 0, 0, 0] for name, *_ in CONFIGS}
    jobs = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for name, kind, cid, level_name, respects in CONFIGS:
            for start, chunk in _chunks(deal_seeds, CHUNK):
                jobs.append(pool.submit(worker, name, kind, cid, level_name,
                                        respects, start, chunk))
        done = 0
        t0 = time.time()
        for fut in cf.as_completed(jobs):
            name, start, per_deal, fb, fs, ifb, ifs = fut.result()
            results[name][start:start + len(per_deal)] = per_deal
            tallies[name][0] += fb
            tallies[name][1] += fs
            if ifb is not None:
                tallies[name][2] += ifb
                tallies[name][3] += ifs
            done += 1
            mem = total_rss_gb()
            print(f"[{done}/{len(jobs)}] {name} deals {start}-"
                  f"{start + len(per_deal) - 1} done | mem={mem:.1f}GB | "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)
            _append_partial(f"{name}@{start}", per_deal)
            if mem > MEM_LIMIT_GB:
                print(f"ABORT: memory {mem:.1f}GB exceeds "
                      f"{MEM_LIMIT_GB}GB limit", flush=True)
                for j in jobs:
                    j.cancel()
                raise MemoryError("memory limit exceeded")
    return results, tallies


def as_rows(results, tallies, deal_seeds, elapsed):
    rows = []
    for name, kind, cid, level_name, respects in CONFIGS:
        per_deal = results[name]
        mean, lo, hi = bootstrap_ci(per_deal)
        fb, fs, ifb, ifs = tallies[name]
        rows.append({"name": name, "per_deal": per_deal, "mean": mean,
                     "lo": lo, "hi": hi, "seconds": elapsed,
                     "fallbacks": fb, "failed_samples": fs,
                     "inner_fallbacks": ifb if kind == "honest" else None,
                     "inner_failed_samples": ifs if kind == "honest" else None})
    return rows


# ---------------------------------------------------------------- probe
def timing_probe():
    """5 deals of honest-FULL (n_outer=50, n_inner=20) vs 3 heuristics."""
    probe_seeds = [900_000 + i for i in range(5)]
    print("[probe] 5 deals x 4 rotations, HonestSearchPlayer(FULL, "
          f"n_outer={N_OUTER}, n_inner={N_INNER})", flush=True)
    t0 = time.time()
    per_deal = rotated_match(
        probe_seeds, honest_factory(999, Level.FULL, True, probe_seeds),
        lambda: HeuristicPlayer())
    elapsed = time.time() - t0
    n_games = 4 * len(probe_seeds)
    s_per_game = elapsed / n_games
    print(f"[probe] mean bot points/deal = {per_deal.mean():.2f}", flush=True)
    print(f"[probe] {elapsed:.1f}s for {n_games} games "
          f"= {s_per_game:.2f}s/game ({elapsed/len(probe_seeds):.2f}s/deal)",
          flush=True)
    return s_per_game


def project_total_seconds(s_per_game, n_deals):
    """3 honest rows dominate cost; the bridge row is cheap (Phase-1 cost,
    already known ~0.63s/game from ablation.txt) and is added in."""
    games = 4 * n_deals
    honest_seconds = 3 * games * s_per_game
    bridge_seconds = games * 0.7   # generous Phase-1-style estimate
    return honest_seconds + bridge_seconds


# ---------------------------------------------------------------- outputs
def write_outputs(rows, n_deals, deal_seeds, s_per_game, note, elapsed,
                  workers, out_name="ablation2.txt", png_name="ablation2.png"):
    os.makedirs(RESULTS, exist_ok=True)
    lines = [
        "open-hearts Phase 2 ablation (does honest search make beliefs pay?)",
        "Lower points per hand is better. 26 points are dealt out per hand,",
        "so 6.5 is the symmetric break-even; a bot below 6.5 is beating the",
        "three heuristic opponents it plays against.",
        "",
        f"deals: {n_deals} (seeds {deal_seeds[0]}..{deal_seeds[-1]}), "
        f"each played 4x with the bot rotated through every seat",
        "identical deals for every configuration; CIs are 10,000 bootstrap",
        "resamples over DEALS (not games).",
        f"parameters: honest rows n_outer={N_OUTER}, n_inner={N_INNER}; "
        f"bridge row n_samples={MAIN_N_SAMPLES} (Phase-1 SearchPlayer, "
        f"config_id={BRIDGE_CONFIG_ID}, identical seeding to ablation.txt's "
        "search-FULL-n100 row)",
        f"timing probe: {s_per_game:.2f}s per honest-FULL game",
        f"wall time: {elapsed:.1f}s with {workers} worker(s)",
    ]
    if note:
        lines += ["", f"NOTE: {note}"]
    lines += [
        "",
        "'fallbk'/'failsmp' are OUTER-level counters (same semantics as "
        "Phase 1: decisions/arrangements where the sampler fell back to the "
        "plain heuristic). 'ifallbk'/'ifailsmp' are the INNER re-determinized "
        "SearchPlayer's counters (n/a for the bridge row).",
        f"{'config':<26}{'mean':>8}{'lo95':>9}{'hi95':>9}{'secs':>10}"
        f"{'fallbk':>8}{'failsmp':>9}{'ifallbk':>9}{'ifailsmp':>9}",
    ]
    for r in rows:
        ifb = "-" if r.get("inner_fallbacks") is None else str(r["inner_fallbacks"])
        ifs = "-" if r.get("inner_failed_samples") is None else str(r["inner_failed_samples"])
        lines.append(f"{r['name']:<26}{r['mean']:>8.3f}{r['lo']:>9.3f}"
                     f"{r['hi']:>9.3f}{r['seconds']:>10.1f}{r['fallbacks']:>8}"
                     f"{r['failed_samples']:>9}{ifb:>9}{ifs:>9}")
    with open(os.path.join(RESULTS, out_name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # bar chart with CI error bars: new rows solid, Phase-1 rows ghosted/hatched
    names = [r["name"] for r in rows] + [n for n, _ in PHASE1_REFERENCE]
    means = [r["mean"] for r in rows] + [m for _, m in PHASE1_REFERENCE]
    err_lo = [r["mean"] - r["lo"] for r in rows] + [0.0] * len(PHASE1_REFERENCE)
    err_hi = [r["hi"] - r["mean"] for r in rows] + [0.0] * len(PHASE1_REFERENCE)
    colors = ["#4C72B0"] * len(rows) + ["#B0B0B0"] * len(PHASE1_REFERENCE)
    hatches = [None] * len(rows) + ["//"] * len(PHASE1_REFERENCE)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, means, yerr=[err_lo, err_hi], capsize=5, color=colors)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_alpha(0.55)
    ax.axhline(6.5, color="grey", linestyle="--", label="6.5 = symmetric break-even")
    ax.set_ylabel("mean points per hand (lower is better)")
    ax.set_title(f"Phase 2 ablation vs Phase 1 (ghosted/hatched), {n_deals} deals x 4 rotations")
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, png_name), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 deals per row, serial, then exit (mechanics check)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if args.smoke:
        smoke_seeds = ra.DEAL_SEEDS[:2]
        with open(PARTIAL, "w") as f:
            f.write(f"# smoke run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        t0 = time.time()
        rows = []
        for name, kind, cid, level_name, respects in CONFIGS:
            rows.append(run_config_serial(name, kind, cid, level_name,
                                          respects, smoke_seeds))
        elapsed = time.time() - t0
        write_outputs(rows, len(smoke_seeds), smoke_seeds, s_per_game=0.0,
                     note="SMOKE RUN: 2 deals/row, mechanics check only -- "
                          "means are not meaningful.", elapsed=elapsed,
                     workers=1, out_name="ablation2_smoke.txt",
                     png_name="ablation2_smoke.png")
        print("[smoke] done", flush=True)
        return

    s_per_game = timing_probe()
    n_deals = N_DEALS_TARGET
    note = None
    projected = project_total_seconds(s_per_game, n_deals)
    print(f"[probe] projected total at {n_deals} deals = "
          f"{projected/3600:.2f}h", flush=True)
    if projected > BUDGET_SECONDS:
        n_deals = 250
        note = (f"projected {projected/3600:.2f}h at 500 deals exceeded the "
                f"2h budget; shrunk to 250 deals per the compute budget rule.")
        print(f"[probe] {note}", flush=True)
        projected = project_total_seconds(s_per_game, n_deals)
        print(f"[probe] projected total at {n_deals} deals = "
              f"{projected/3600:.2f}h", flush=True)
        if projected > BUDGET_SECONDS:
            print(f"\nSTOP: even 250 deals projects {projected/3600:.2f}h, "
                  "over the 2h budget. Per the plan, the lead must consult "
                  "the owner before running. Nothing has been run.",
                  flush=True)
            return

    deal_seeds = ra.DEAL_SEEDS[:n_deals]
    with open(PARTIAL, "w") as f:
        f.write(f"# run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t0 = time.time()
    results, tallies = run_all(deal_seeds, args.workers)
    elapsed = time.time() - t0
    rows = as_rows(results, tallies, deal_seeds, elapsed)
    write_outputs(rows, n_deals, deal_seeds, s_per_game, note, elapsed,
                 args.workers)
    print(f"[done] wall time {elapsed/60:.1f} min with {args.workers} workers",
          flush=True)


if __name__ == "__main__":
    main()
