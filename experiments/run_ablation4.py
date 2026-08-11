"""Phase 4 headline ablation (does a learned value net beat heuristic
playouts, in situ, inside honest search?).

Rows, all vs 3 plain HeuristicPlayer opponents, deal seeds 100000+ (identical
to Phases 1-3), paired per-deal bootstrap against row 1:

    1. honest-CHOICE        Bridge row. Bitwise-identical construction to the
                             Phase-3 Task-9 run that measured 2.87:
                             HonestSearchPlayer(Level.FULL, n_outer=50,
                             n_inner=20), OUTER worlds drawn from
                             WeightedPosterior(Level.FULL, HeuristicPlayer(),
                             epsilon=0, n_worlds=n_outer). Same config_id
                             (310) as run_ablation2.py's --choice-only row,
                             so the per-game rng stream -- and therefore the
                             per-deal values -- reproduce exactly.
    2. value-CHOICE-h1       ValueSearchPlayer(horizon=1), same n_outer=50 /
                             n_inner=20, same CHOICE posterior construction
                             (fresh n_worlds tied to this row's n_outer).
                             Tests the pure evaluator swap at equal worlds.
    3. value-CHOICE-h1-eqt   horizon=1, n_outer/n_inner scaled up
                             (default 165/66, ~sqrt(11) each) to match row 1's
                             wall clock -- the fair equal-COMPUTE fight this
                             ablation is really about. CLI-overridable via
                             --eqt-outer/--eqt-inner so the lead can retune
                             after the --probe timing numbers.
    4. value-CHOICE-h0-eqt   horizon=0, labelled "determinized+net" in every
                             output (h0 never reaches an imagined own
                             decision, so re-determinization -- and n_inner --
                             never fire; the row is a determinized playout
                             with a learned evaluator, not honest search).
                             n_outer CLI-overridable via --h0-outer, default
                             cap 1000 (Phase-1's world-count saturation).
    5. heuristic-mirror      4 plain HeuristicPlayers -- the 6.5 alarm row.

Rows 1-4 all draw their OUTER worlds from the same CHOICE posterior
construction as run_ablation2.py's honest-CHOICE row (WeightedPosterior,
Level.FULL proposal, epsilon=0 -- exact against these HeuristicPlayer
opponents, not an approximation), each with n_worlds tied to that row's own
n_outer (one posterior world per outer world, same convention as Phase 3).

Determinism / opponent contract: identical to run_ablation.py /
run_ablation2.py -- opponents are plain `HeuristicPlayer()` (free to
construct, no seed needed), `bot_factory()` is called once per game in
`rotated_match`'s documented iteration order (seeds outer, rotations inner),
and per-game rngs derive from `run_ablation._game_seed(config_id, deal_seed,
rotation)`.

Budget rule (PHASE4_PLAN Global Constraints): --probe first (5 deals, ALL
rows). Row 1 and row 3's s/game feed the equal-time tuning (the lead adjusts
--eqt-outer/--eqt-inner so row 3's projected s/game is within +-20% of row
1's before the full run). Budget projection is total worker-seconds DIVIDED
BY the worker count vs the 2h cap -- ablation2's main() forgot this division;
this script does not repeat that bug. Target 500 deals; if projected over 2h,
drop to 250 and say so; if 250 is still over 2h, STOP and do not run.

Usage:
    python experiments/run_ablation4.py --smoke   # 2 deals/row, serial, exit
    python experiments/run_ablation4.py --probe   # 5 deals/row, timing only
    python experiments/run_ablation4.py           # probe + full run (LEAD ONLY)
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
from openhearts.belief.weighted import WeightedPosterior  # noqa: E402
from openhearts.eval.harness import rotated_match  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402
from openhearts.search.valuesearch import ValueSearchPlayer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
PARTIAL = os.path.join(RESULTS, "ablation4_partial.txt")

N_DEALS_TARGET = 500
BUDGET_SECONDS = 2 * 3600
WORKERS = 8
CHUNK = 25
MEM_LIMIT_GB = 100.0

# Bridge row: MUST reuse config_id=310, the exact id run_ablation2.py's
# --choice-only row used, or the per-game rng stream (and hence the per-deal
# values) will not reproduce 2.87 bitwise.
BRIDGE_CONFIG_ID = 310
BRIDGE_N_OUTER = 50
BRIDGE_N_INNER = 20

VALUE_H1_CONFIG_ID = 320
VALUE_H1_EQT_CONFIG_ID = 321
VALUE_H0_EQT_CONFIG_ID = 322

CHOICE_EPSILON = 0.0
CHOICE_MAX_DRAWS = 50_000  # same cap the Phase-3 choice row used

EQT_OUTER_DEFAULT = 165
EQT_INNER_DEFAULT = 66
H0_OUTER_DEFAULT = 1000
H0_N_INNER = 20  # inert at horizon 0 -- never consulted, kept for symmetry

# Phase-2/3 rows, hardcoded from their published results files, for the
# ghosted comparison bars.
GHOST_REFERENCE = [
    ("phase1-FULL", 3.27),
    ("honest-FULL", 3.25),
    ("honest-CHOICE (phase 3)", 2.87),
]


def _posterior_factory(n_worlds):
    """OUTER worlds via WeightedPosterior at eps=0, tied to this row's
    n_outer -- identical construction to run_ablation2.py's honest-CHOICE
    row, just parameterized so each row's posterior samples the right
    number of worlds.
    """
    def factory(view, rng):
        return WeightedPosterior.from_view(
            view, Level.FULL, HeuristicPlayer(), CHOICE_EPSILON,
            n_worlds, rng=rng, max_draws=CHOICE_MAX_DRAWS, keep_worlds=True)
    return factory


# Row spec: (name, kind, config_id, n_outer, n_inner, horizon)
# kind in {"honest", "value", "heuristic"}
def _row_specs(eqt_outer, eqt_inner, h0_outer):
    return [
        ("honest-CHOICE", "honest", BRIDGE_CONFIG_ID,
         BRIDGE_N_OUTER, BRIDGE_N_INNER, None),
        ("value-CHOICE-h1", "value", VALUE_H1_CONFIG_ID,
         BRIDGE_N_OUTER, BRIDGE_N_INNER, 1),
        ("value-CHOICE-h1-eqt", "value", VALUE_H1_EQT_CONFIG_ID,
         eqt_outer, eqt_inner, 1),
        ("value-CHOICE-h0-eqt", "value", VALUE_H0_EQT_CONFIG_ID,
         h0_outer, H0_N_INNER, 0),
        ("heuristic-mirror", "heuristic", None, None, None, None),
    ]


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


def _make_bot_factory(kind, config_id, n_outer, n_inner, horizon,
                      deal_seeds, tally):
    """Built with primitives only (picklable to a ProcessPoolExecutor)."""
    counter = itertools.count()

    def factory():
        i = next(counter)
        seed, rotation = deal_seeds[i // 4], i % 4
        if kind == "heuristic":
            return HeuristicPlayer()
        rng = ra._game_seed(config_id, seed, rotation)
        pf = _posterior_factory(n_outer)
        if kind == "honest":
            bot = HonestSearchPlayer(Level.FULL, n_outer, n_inner, rng,
                                     sampler_respects_voids=True,
                                     posterior_factory=pf)
        elif kind == "value":
            bot = ValueSearchPlayer(Level.FULL, n_outer, n_inner, rng,
                                    sampler_respects_voids=True,
                                    posterior_factory=pf, horizon=horizon)
        else:
            raise ValueError(kind)
        if tally is not None:
            tally.setdefault("bots", []).append(bot)
        return bot
    return factory


def _counters(kind, tally):
    """Sum (fallbacks, failed_samples, inner_fallbacks, inner_failed_samples,
    posterior [collapses, decisions_served, worlds_supplied])."""
    bots = tally.get("bots", []) if tally is not None else []
    if kind == "heuristic":
        return 0, 0, None, None, None
    fb = sum(b.fallbacks for b in bots)
    fs = sum(b.failed_samples for b in bots)
    ifb = sum(b.inner_fallbacks for b in bots) if hasattr(bots[0], "inner_fallbacks") else 0
    ifs = sum(b.inner_failed_samples for b in bots) if hasattr(bots[0], "inner_failed_samples") else 0
    pc = [sum(b.posterior_collapses for b in bots),
          sum(b.posterior_decisions for b in bots),
          sum(b.posterior_worlds for b in bots)]
    return fb, fs, ifb, ifs, pc


def run_config_serial(name, kind, config_id, n_outer, n_inner, horizon,
                      deal_seeds):
    """Serial (used by --smoke and --probe); mirrors run_ablation2's helper."""
    tally = {}
    bot_factory = _make_bot_factory(kind, config_id, n_outer, n_inner,
                                    horizon, deal_seeds, tally)
    print(f"[config] {name}: {len(deal_seeds)} deals x 4 rotations", flush=True)
    t0 = time.time()
    per_deal = rotated_match(deal_seeds, bot_factory,
                             lambda: HeuristicPlayer(), on_deal=_progress(name))
    elapsed = time.time() - t0
    mean, lo, hi = bootstrap_ci(per_deal)
    fb, fs, ifb, ifs, pc = _counters(kind, tally)
    print(f"[config] {name}: mean={mean:.3f} CI=({lo:.3f},{hi:.3f}) "
          f"in {elapsed:.1f}s fallbacks={fb} failed={fs} "
          f"inner_fallbacks={ifb} inner_failed={ifs} "
          f"posterior={pc}", flush=True)
    return {"name": name, "per_deal": per_deal, "mean": mean, "lo": lo,
            "hi": hi, "seconds": elapsed, "fallbacks": fb, "failed_samples": fs,
            "inner_fallbacks": ifb, "inner_failed_samples": ifs,
            "posterior": pc}


# ---------------------------------------------------------------- parallel
def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i:i + size]


def worker(name, kind, config_id, n_outer, n_inner, horizon, start,
          chunk_seeds):
    """Run one chunk of deals for one configuration in a worker process."""
    tally = {}
    bot_factory = _make_bot_factory(kind, config_id, n_outer, n_inner,
                                    horizon, chunk_seeds, tally)
    per_deal = rotated_match(chunk_seeds, bot_factory, lambda: HeuristicPlayer())
    fb, fs, ifb, ifs, pc = _counters(kind, tally)
    return (name, start, per_deal, fb, fs, ifb, ifs, pc)


def total_rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


def run_all(deal_seeds, workers, specs):
    import concurrent.futures as cf
    results = {name: np.zeros(len(deal_seeds)) for name, *_ in specs}
    tallies = {name: [0, 0, 0, 0, [0, 0, 0]] for name, *_ in specs}
    jobs = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for name, kind, cid, n_outer, n_inner, horizon in specs:
            for start, chunk in _chunks(deal_seeds, CHUNK):
                jobs.append(pool.submit(worker, name, kind, cid, n_outer,
                                        n_inner, horizon, start, chunk))
        done = 0
        t0 = time.time()
        for fut in cf.as_completed(jobs):
            name, start, per_deal, fb, fs, ifb, ifs, pc = fut.result()
            results[name][start:start + len(per_deal)] = per_deal
            tallies[name][0] += fb
            tallies[name][1] += fs
            if ifb is not None:
                tallies[name][2] += ifb
                tallies[name][3] += ifs
            if pc is not None:
                for k in range(3):
                    tallies[name][4][k] += pc[k]
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


def as_rows(results, tallies, specs):
    rows = []
    for name, kind, cid, n_outer, n_inner, horizon in specs:
        per_deal = results[name]
        mean, lo, hi = bootstrap_ci(per_deal)
        fb, fs, ifb, ifs, pc = tallies[name]
        inner = kind in ("honest", "value")
        rows.append({"name": name, "per_deal": per_deal, "mean": mean,
                     "lo": lo, "hi": hi,
                     "fallbacks": fb, "failed_samples": fs,
                     "inner_fallbacks": ifb if inner else None,
                     "inner_failed_samples": ifs if inner else None,
                     "posterior": pc if kind in ("honest", "value") else None})
    return rows


# ---------------------------------------------------------------- probe
def timing_probe(specs, n_probe_deals=5):
    """5 deals of every row, serial, s/game each -- rows 1 and 3 are the ones
    the equal-time tuning cares about, but all five are reported."""
    probe_seeds = [900_000 + i for i in range(n_probe_deals)]
    s_per_game = {}
    rows = {}
    for name, kind, cid, n_outer, n_inner, horizon in specs:
        row = run_config_serial(name, kind, cid, n_outer, n_inner, horizon,
                                probe_seeds)
        n_games = 4 * n_probe_deals
        spg = row["seconds"] / n_games
        s_per_game[name] = spg
        rows[name] = row
        print(f"[probe] {name}: {spg:.3f}s/game "
              f"({row['seconds']:.1f}s for {n_games} games)", flush=True)
    return s_per_game, rows, probe_seeds


def project_total_seconds(s_per_game, n_deals):
    games = 4 * n_deals
    return sum(spg * games for spg in s_per_game.values())


def print_budget_projection(s_per_game, n_deals, workers):
    total = project_total_seconds(s_per_game, n_deals)
    projected_wall = total / workers
    print("=" * 72, flush=True)
    print(f"[BUDGET] projected total worker-seconds at {n_deals} deals = "
          f"{total:.0f}s ({total/3600:.2f}h serial)", flush=True)
    print(f"[BUDGET] DIVIDED BY {workers} workers -> wall clock = "
          f"{projected_wall:.0f}s ({projected_wall/3600:.2f}h) "
          f"vs {BUDGET_SECONDS/3600:.1f}h cap", flush=True)
    print("=" * 72, flush=True)
    return total, projected_wall


def eqt_ratio_report(s_per_game):
    bridge = s_per_game.get("honest-CHOICE")
    eqt = s_per_game.get("value-CHOICE-h1-eqt")
    if bridge and eqt:
        ratio = eqt / bridge
        print(f"[probe] equal-time check: value-CHOICE-h1-eqt / honest-CHOICE "
              f"s/game ratio = {ratio:.2f} (target 1.00 +-0.20; adjust "
              "--eqt-outer/--eqt-inner if outside that band)", flush=True)


# ---------------------------------------------------------------- outputs
def paired_bootstrap_ci(diffs, n_boot=10_000, rng=None):
    """Paired bootstrap over per-deal (row - bridge) differences: resampling
    deals resamples the paired diff at each deal, which is exactly what
    bootstrap_ci already does over a 1-D array."""
    return bootstrap_ci(diffs, n_boot=n_boot, rng=rng)


def write_outputs(rows, n_deals, deal_seeds, s_per_game, note, elapsed,
                  workers, eqt_outer, eqt_inner, h0_outer,
                  out_name="ablation4.txt", png_name="ablation4.png"):
    os.makedirs(RESULTS, exist_ok=True)
    by_name = {r["name"]: r for r in rows}
    bridge = by_name["honest-CHOICE"]

    lines = [
        "open-hearts Phase 4 headline ablation (learned value net vs "
        "heuristic playouts, in situ, inside honest search).",
        "Lower points per hand is better. 26 points are dealt out per hand,",
        "so 6.5 is the symmetric break-even; a bot below 6.5 is beating the",
        "three heuristic opponents it plays against.",
        "",
        f"deals: {n_deals} (seeds {deal_seeds[0]}..{deal_seeds[-1]}), "
        "each played 4x with the bot rotated through every seat; identical "
        "deals for every configuration. CIs are 10,000 bootstrap resamples "
        "over DEALS (not games).",
        "",
        "parameters:",
        f"  honest-CHOICE (bridge, config_id={BRIDGE_CONFIG_ID}): "
        f"HonestSearchPlayer(Level.FULL, n_outer={BRIDGE_N_OUTER}, "
        f"n_inner={BRIDGE_N_INNER}), CHOICE posterior (WeightedPosterior, "
        f"Level.FULL proposal, HeuristicPlayer, epsilon={CHOICE_EPSILON}, "
        f"n_worlds={BRIDGE_N_OUTER}, max_draws={CHOICE_MAX_DRAWS}). "
        "MUST reproduce 2.87 bitwise (same config_id=310 as "
        "run_ablation2.py's --choice-only row).",
        f"  value-CHOICE-h1 (config_id={VALUE_H1_CONFIG_ID}): "
        f"ValueSearchPlayer(horizon=1, n_outer={BRIDGE_N_OUTER}, "
        f"n_inner={BRIDGE_N_INNER}), same CHOICE posterior construction.",
        f"  value-CHOICE-h1-eqt (config_id={VALUE_H1_EQT_CONFIG_ID}): "
        f"ValueSearchPlayer(horizon=1, n_outer={eqt_outer}, "
        f"n_inner={eqt_inner}) -- scaled to match the bridge row's wall "
        "clock; the equal-COMPUTE headline row.",
        f"  value-CHOICE-h0-eqt / \"determinized+net\" "
        f"(config_id={VALUE_H0_EQT_CONFIG_ID}): ValueSearchPlayer(horizon=0, "
        f"n_outer={h0_outer}, n_inner={H0_N_INNER} but INERT -- horizon 0 "
        "never reaches an imagined own decision, so re-determinization "
        "never fires and n_inner is not consulted). This row is a "
        "determinized playout with a learned evaluator, not honest search.",
        "  heuristic-mirror: 4 plain HeuristicPlayers (the 6.5 alarm row).",
        "",
        f"timing probe s/game per row: " +
        ", ".join(f"{n}={s:.3f}" for n, s in s_per_game.items()),
        f"wall time: {elapsed:.1f}s with {workers} worker(s)",
    ]
    if note:
        lines += ["", f"NOTE: {note}"]

    lines += [
        "",
        "'fallbk'/'failsmp' are OUTER-level counters (decisions/arrangements "
        "where the sampler fell back to the plain heuristic). "
        "'ifallbk'/'ifailsmp' are the INNER re-determinized search's "
        "counters (n/a for heuristic-mirror; inert-but-present for h0-eqt). "
        "'pcollapse' counts decisions where NO candidate world survived "
        "choice filtering and the row fell back to the constraint sampler "
        "(n/a for heuristic-mirror).",
        f"{'config':<26}{'mean':>8}{'lo95':>9}{'hi95':>9}"
        f"{'fallbk':>8}{'failsmp':>9}{'ifallbk':>9}{'ifailsmp':>9}"
        f"{'pcollapse':>11}",
    ]
    for r in rows:
        ifb = "-" if r.get("inner_fallbacks") is None else str(r["inner_fallbacks"])
        ifs = "-" if r.get("inner_failed_samples") is None else str(r["inner_failed_samples"])
        post = r.get("posterior")
        pc = "-" if post is None else str(post[0])
        lines.append(f"{r['name']:<26}{r['mean']:>8.3f}{r['lo']:>9.3f}"
                     f"{r['hi']:>9.3f}{r['fallbacks']:>8}"
                     f"{r['failed_samples']:>9}{ifb:>9}{ifs:>9}{pc:>11}")

    lines += ["", "paired per-deal diffs vs honest-CHOICE (row - bridge; "
              "negative = row beats the bridge), 95% paired bootstrap CI:",
              f"{'config':<26}{'mean_diff':>11}{'lo95':>9}{'hi95':>9}"]
    for r in rows:
        if r["name"] == "honest-CHOICE":
            continue
        diffs = r["per_deal"] - bridge["per_deal"]
        dmean, dlo, dhi = paired_bootstrap_ci(diffs)
        lines.append(f"{r['name']:<26}{dmean:>11.3f}{dlo:>9.3f}{dhi:>9.3f}")

    with open(os.path.join(RESULTS, out_name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # bar chart with CI error bars: new rows solid, Phase-2/3 rows ghosted
    names = [r["name"] for r in rows] + [n for n, _ in GHOST_REFERENCE]
    means = [r["mean"] for r in rows] + [m for _, m in GHOST_REFERENCE]
    err_lo = [r["mean"] - r["lo"] for r in rows] + [0.0] * len(GHOST_REFERENCE)
    err_hi = [r["hi"] - r["mean"] for r in rows] + [0.0] * len(GHOST_REFERENCE)
    colors = ["#4C72B0"] * len(rows) + ["#B0B0B0"] * len(GHOST_REFERENCE)
    hatches = [None] * len(rows) + ["//"] * len(GHOST_REFERENCE)
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, means, yerr=[err_lo, err_hi], capsize=5, color=colors)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_alpha(0.55)
    ax.axhline(6.5, color="grey", linestyle="--", label="6.5 = symmetric break-even")
    ax.set_ylabel("mean points per hand (lower is better)")
    ax.set_title(f"Phase 4 ablation vs Phase 2/3 (ghosted/hatched), "
                 f"{n_deals} deals x 4 rotations")
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, png_name), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 deals per row, serial, then exit (mechanics check)")
    ap.add_argument("--probe", action="store_true",
                    help="5 deals per row, serial, timing only, then exit")
    ap.add_argument("--deals", type=int, default=N_DEALS_TARGET)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--eqt-outer", type=int, default=EQT_OUTER_DEFAULT)
    ap.add_argument("--eqt-inner", type=int, default=EQT_INNER_DEFAULT)
    ap.add_argument("--h0-outer", type=int, default=H0_OUTER_DEFAULT)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    specs = _row_specs(args.eqt_outer, args.eqt_inner, args.h0_outer)

    if args.smoke:
        smoke_seeds = ra.DEAL_SEEDS[:2]
        with open(PARTIAL, "w") as f:
            f.write(f"# smoke run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        t0 = time.time()
        rows = []
        for name, kind, cid, n_outer, n_inner, horizon in specs:
            row = run_config_serial(name, kind, cid, n_outer, n_inner,
                                    horizon, smoke_seeds)
            _append_partial(row["name"], row["per_deal"])
            rows.append(row)
        elapsed = time.time() - t0
        s_per_game = {r["name"]: (r["seconds"] / (4 * len(smoke_seeds)))
                      for r in rows}
        write_outputs(rows, len(smoke_seeds), smoke_seeds, s_per_game,
                     note="SMOKE RUN: 2 deals/row, mechanics check only -- "
                          "means are not meaningful.", elapsed=elapsed,
                     workers=1, eqt_outer=args.eqt_outer,
                     eqt_inner=args.eqt_inner, h0_outer=args.h0_outer,
                     out_name="ablation4_smoke.txt",
                     png_name="ablation4_smoke.png")
        print("[smoke] done", flush=True)
        return

    if args.probe:
        s_per_game, _rows, probe_seeds = timing_probe(specs)
        eqt_ratio_report(s_per_game)
        print_budget_projection(s_per_game, args.deals, args.workers)
        print("[probe] done (no full run performed)", flush=True)
        return

    s_per_game, _rows, probe_seeds = timing_probe(specs)
    eqt_ratio_report(s_per_game)
    n_deals = args.deals
    note = None
    total, projected_wall = print_budget_projection(s_per_game, n_deals,
                                                     args.workers)
    if projected_wall > BUDGET_SECONDS:
        n_deals = 250
        total, projected_wall = print_budget_projection(s_per_game, n_deals,
                                                         args.workers)
        note = (f"projected {projected_wall/3600:.2f}h at {args.deals} deals "
                f"exceeded the 2h budget; shrunk to 250 deals per the "
                "compute budget rule.")
        print(f"[probe] {note}", flush=True)
        if projected_wall > BUDGET_SECONDS:
            print(f"\nSTOP: even 250 deals projects "
                  f"{projected_wall/3600:.2f}h, over the 2h budget. Per the "
                  "plan the lead must consult the owner. Nothing has been "
                  "run.", flush=True)
            return

    deal_seeds = ra.DEAL_SEEDS[:n_deals]
    with open(PARTIAL, "w") as f:
        f.write(f"# run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t0 = time.time()
    results, tallies = run_all(deal_seeds, args.workers, specs)
    elapsed = time.time() - t0
    rows = as_rows(results, tallies, specs)
    write_outputs(rows, n_deals, deal_seeds, s_per_game, note, elapsed,
                 args.workers, args.eqt_outer, args.eqt_inner, args.h0_outer)
    print(f"[done] wall time {elapsed/60:.1f} min with {args.workers} workers",
          flush=True)


if __name__ == "__main__":
    main()
