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

Phase 3 adds a fifth row, run SEPARATELY via `--choice-only` (config_id 310,
output results/ablation3.txt):

    5. honest-CHOICE          (HonestSearchPlayer, Level.FULL, n_outer=50,
                                n_inner=20, OUTER worlds drawn from
                                WeightedPosterior(epsilon=0))

It is not part of the default four-row run: those numbers are already
published in results/ablation2.txt, and rerunning them to add one row would
waste the budget. `--choice-only` runs the new row alone on the same deal
seeds and writes ablation3.txt, which compares it against the Phase-2 means
hardcoded in PHASE2_REFERENCE below.

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
from openhearts.belief.weighted import WeightedPosterior  # noqa: E402
from openhearts.eval.harness import rotated_match  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.search.decision import SearchPlayer  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
PARTIAL = os.path.join(RESULTS, "ablation2_partial.txt")
PARTIAL3 = os.path.join(RESULTS, "ablation3_partial.txt")
# Which partial file _append_partial writes to; --choice-only repoints it so
# a Phase-3 run can never clobber the Phase-2 partial.
ACTIVE_PARTIAL = PARTIAL

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

# ------------------------------------------------------- Phase 3 choice row
CHOICE_CONFIG_ID = 310
CHOICE_EPSILON = 0.0
CHOICE_N_WORLDS = N_OUTER      # one posterior world per outer world
CHOICE_MAX_DRAWS = 50_000      # same cap Task 6's survival measurement used
CHOICE_CONFIG = ("honest-CHOICE", "choice", CHOICE_CONFIG_ID, "FULL", True)

# Phase-2 numbers hardcoded from results/ablation2.txt (500 deals, identical
# seeds), so ablation3.txt is self-contained and the honest-CHOICE row can be
# read against them without rerunning anything.
PHASE2_REFERENCE = [
    ("honest-UNIFORM-noleak", 3.469, 3.270, 3.666),
    ("honest-VOIDS", 3.343, 3.151, 3.538),
    ("honest-FULL", 3.253, 3.070, 3.447),
    ("phase1-FULL-bridge", 3.267, 3.091, 3.447),
]


def _posterior_factory():
    """OUTER worlds for the honest-CHOICE row: WeightedPosterior at eps=0.

    epsilon=0 is the CORRECT model here, not an approximation: the three
    opponents in this row are plain `HeuristicPlayer()`s, exactly the policy
    the posterior audits against, and they never deviate. Rejection sampling
    against the true policy is therefore exact, and every surviving world has
    weight 1.0. (epsilon > 0 belongs to the robustness experiment, where the
    opponents really do deviate.)

    Level.FULL for the proposal table: the choice row is the full-pipeline
    row, so both stages get the best available belief.

    The posterior sees only `view`, and draws from the player's own rng, so
    the row stays deterministic given its config/deal/rotation seed.
    """
    def factory(view, rng):
        return WeightedPosterior.from_view(
            view, Level.FULL, HeuristicPlayer(), CHOICE_EPSILON,
            CHOICE_N_WORLDS, rng=rng, max_draws=CHOICE_MAX_DRAWS,
            keep_worlds=True)
    return factory


def choice_factory(config_id, deal_seeds, tally=None):
    """honest-CHOICE bots, seeded exactly like the other honest rows."""
    counter = itertools.count()
    def factory():
        i = next(counter)
        seed, rotation = deal_seeds[i // 4], i % 4
        bot = HonestSearchPlayer(Level.FULL, N_OUTER, N_INNER,
                                 ra._game_seed(config_id, seed, rotation),
                                 sampler_respects_voids=True,
                                 posterior_factory=_posterior_factory())
        if tally is not None:
            tally.setdefault("bots", []).append(bot)
        return bot
    return factory


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
    with open(ACTIVE_PARTIAL, "a") as f:
        f.write(name + " " + " ".join(f"{v:.6f}" for v in values) + "\n")
        f.flush()


def _make_bot_factory(kind, config_id, level_name, respects, deal_seeds, tally):
    if kind == "honest":
        return honest_factory(config_id, Level[level_name], respects,
                              deal_seeds, tally)
    if kind == "bridge":
        return bridge_factory(config_id, deal_seeds, tally)
    if kind == "choice":
        # Built HERE, inside the worker process: `worker()` receives only
        # primitives precisely because closures do not survive pickling to a
        # ProcessPoolExecutor.
        return choice_factory(config_id, deal_seeds, tally)
    raise ValueError(kind)


def _counters(kind, tally):
    """Sum (fallbacks, failed_samples, inner_fallbacks, inner_failed_samples,
    posterior).

    `posterior` is None off the choice row, else the 3-list
    [collapses, decisions_served, worlds_supplied] -- the last two give the
    realised worlds/decision, which is < n_outer whenever the posterior ran
    out of draws before filling its quota.
    """
    bots = tally.get("bots", []) if tally is not None else []
    fb = sum(b.fallbacks for b in bots)
    fs = sum(b.failed_samples for b in bots)
    if kind in ("honest", "choice"):
        ifb = sum(b.inner_fallbacks for b in bots)
        ifs = sum(b.inner_failed_samples for b in bots)
    else:
        ifb = ifs = None
    # Decisions where NO candidate world survived choice filtering and the
    # player fell back to the constraint sampler. Counted and printed, never
    # silent (PHASE23_PLAN truth-safety rule).
    if kind == "choice":
        pc = [sum(b.posterior_collapses for b in bots),
              sum(b.posterior_decisions for b in bots),
              sum(b.posterior_worlds for b in bots)]
    else:
        pc = None
    return fb, fs, ifb, ifs, pc


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
    fb, fs, ifb, ifs, pc = _counters(kind, tally)
    print(f"[config] {name}: mean={mean:.3f} CI=({lo:.3f},{hi:.3f}) "
          f"in {elapsed:.1f}s fallbacks={fb} failed={fs} "
          f"inner_fallbacks={ifb} inner_failed={ifs} "
          f"posterior={pc}", flush=True)
    _append_partial(name, per_deal)
    return {"name": name, "per_deal": per_deal, "mean": mean, "lo": lo,
            "hi": hi, "seconds": elapsed, "fallbacks": fb, "failed_samples": fs,
            "inner_fallbacks": ifb, "inner_failed_samples": ifs,
            "posterior": pc}


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
    fb, fs, ifb, ifs, pc = _counters(kind, tally)
    return (name, start, per_deal, fb, fs, ifb, ifs, pc)


def total_rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


def run_all(deal_seeds, workers, configs=None):
    configs = CONFIGS if configs is None else configs
    import concurrent.futures as cf
    results = {name: np.zeros(len(deal_seeds)) for name, *_ in configs}
    tallies = {name: [0, 0, 0, 0, [0, 0, 0]] for name, *_ in configs}
    jobs = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for name, kind, cid, level_name, respects in configs:
            for start, chunk in _chunks(deal_seeds, CHUNK):
                jobs.append(pool.submit(worker, name, kind, cid, level_name,
                                        respects, start, chunk))
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


def as_rows(results, tallies, deal_seeds, elapsed, configs=None):
    configs = CONFIGS if configs is None else configs
    rows = []
    for name, kind, cid, level_name, respects in configs:
        per_deal = results[name]
        mean, lo, hi = bootstrap_ci(per_deal)
        fb, fs, ifb, ifs, pc = tallies[name]
        inner = kind in ("honest", "choice")
        rows.append({"name": name, "per_deal": per_deal, "mean": mean,
                     "lo": lo, "hi": hi, "seconds": elapsed,
                     "fallbacks": fb, "failed_samples": fs,
                     "inner_fallbacks": ifb if inner else None,
                     "inner_failed_samples": ifs if inner else None,
                     "posterior": pc if kind == "choice" else None})
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


# ------------------------------------------------- Phase 3 choice-row output
def choice_timing_probe():
    """5 deals of honest-CHOICE vs 3 heuristics (same shape as timing_probe)."""
    probe_seeds = [900_000 + i for i in range(5)]
    print("[probe] 5 deals x 4 rotations, honest-CHOICE (FULL, "
          f"n_outer={N_OUTER}, n_inner={N_INNER}, posterior eps="
          f"{CHOICE_EPSILON}, n_worlds={CHOICE_N_WORLDS}, "
          f"max_draws={CHOICE_MAX_DRAWS})", flush=True)
    tally = {}
    t0 = time.time()
    per_deal = rotated_match(probe_seeds,
                             choice_factory(999, probe_seeds, tally),
                             lambda: HeuristicPlayer())
    elapsed = time.time() - t0
    n_games = 4 * len(probe_seeds)
    s_per_game = elapsed / n_games
    _fb, _fs, _ifb, _ifs, pc = _counters("choice", tally)
    print(f"[probe] mean bot points/deal = {per_deal.mean():.2f}", flush=True)
    print(f"[probe] {elapsed:.1f}s for {n_games} games = {s_per_game:.2f}s/game"
          f" | posterior_collapses={pc}", flush=True)
    return s_per_game


def write_choice_outputs(row, n_deals, deal_seeds, s_per_game, note, elapsed,
                         workers, out_name="ablation3.txt",
                         png_name="ablation3.png"):
    """ablation3.txt: the honest-CHOICE row against the stored Phase-2 rows."""
    os.makedirs(RESULTS, exist_ok=True)
    lines = [
        "open-hearts Phase 3 full-pipeline row (choice-aware belief -> honest",
        "search -> points). Lower points per hand is better; 6.5 is the",
        "symmetric break-even against three heuristic opponents.",
        "",
        f"deals: {n_deals} (seeds {deal_seeds[0]}..{deal_seeds[-1]}), each "
        "played 4x with the bot rotated through every seat -- the SAME deal "
        "seeds as ablation.txt and ablation2.txt, so the rows below are "
        "directly comparable.",
        "CIs are 10,000 bootstrap resamples over DEALS (not games).",
        "",
        f"honest-CHOICE (config_id={CHOICE_CONFIG_ID}): HonestSearchPlayer("
        f"Level.FULL, n_outer={N_OUTER}, n_inner={N_INNER}) whose OUTER worlds "
        f"come from WeightedPosterior(Level.FULL, HeuristicPlayer, "
        f"epsilon={CHOICE_EPSILON}, n_worlds={CHOICE_N_WORLDS}, "
        f"max_draws={CHOICE_MAX_DRAWS}) instead of the raw constraint sampler.",
        "epsilon=0 is the CORRECT model for this row, not an approximation: "
        "the three opponents ARE plain HeuristicPlayers, exactly the policy "
        "the posterior audits against, and they never deviate. Every "
        "surviving world therefore has weight 1.0 and the kept worlds are an "
        "exact rejection sample. (epsilon > 0 is the robustness experiment's "
        "question, not this row's.)",
        "INNER re-determinization stays constraint-based (BeliefTable + "
        "sampler). That is a correctness choice, not a budget cut: the plays "
        "'observed' inside an imagined world were invented by the playout "
        "itself, so auditing them would only measure how well an imagined "
        "world explains its own imagined moves.",
        "",
        f"parameters: workers={workers}, wall time {elapsed:.1f}s, "
        f"timing probe {s_per_game:.2f}s per honest-CHOICE game",
    ]
    if note:
        lines += ["", f"NOTE: {note}"]
    lines += [
        "",
        "'pcollapse' counts decisions where NO candidate world survived choice "
        "filtering within max_draws and the player fell back to the plain "
        "constraint sampler for that one decision. The plan's truth-safety "
        "rule forbids a SILENT fallback; this one is counted and printed here. "
        "A large number means the row is partly a plain honest-FULL row and "
        "must be read that way.",
        "Rows marked (phase 2) are quoted verbatim from results/ablation2.txt "
        "(500 deals, identical seeds); they were not rerun.",
        "",
        f"{'config':<30}{'mean':>8}{'lo95':>9}{'hi95':>9}{'secs':>10}"
        f"{'fallbk':>8}{'failsmp':>9}{'ifallbk':>9}{'ifailsmp':>9}"
        f"{'pcollapse':>11}",
    ]
    post = row.get("posterior") or [None, 0, 0]
    pc = post[0]
    served, supplied = post[1], post[2]
    lines.append(
        f"{row['name']:<30}{row['mean']:>8.3f}{row['lo']:>9.3f}"
        f"{row['hi']:>9.3f}{row['seconds']:>10.1f}{row['fallbacks']:>8}"
        f"{row['failed_samples']:>9}{row['inner_fallbacks']:>9}"
        f"{row['inner_failed_samples']:>9}"
        f"{('-' if pc is None else str(pc)):>11}")
    for name, mean, lo, hi in PHASE2_REFERENCE:
        lines.append(f"{name + ' (phase 2)':<30}{mean:>8.3f}{lo:>9.3f}"
                     f"{hi:>9.3f}{'-':>10}{'-':>8}{'-':>9}{'-':>9}{'-':>9}"
                     f"{'-':>11}")
    if served:
        lines += ["",
                  f"posterior served {served} decisions with {supplied} "
                  f"worlds = {supplied / served:.1f} worlds/decision (quota "
                  f"n_worlds={CHOICE_N_WORLDS}); a figure well below the "
                  f"quota means max_draws, not the choice evidence, was the "
                  f"binding constraint."]
    with open(os.path.join(RESULTS, out_name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    names = [row["name"]] + [n for n, *_ in PHASE2_REFERENCE]
    means = [row["mean"]] + [m for _, m, _, _ in PHASE2_REFERENCE]
    err_lo = [row["mean"] - row["lo"]] + [m - lo for _, m, lo, _ in PHASE2_REFERENCE]
    err_hi = [row["hi"] - row["mean"]] + [hi - m for _, m, _, hi in PHASE2_REFERENCE]
    colors = ["#4C72B0"] + ["#B0B0B0"] * len(PHASE2_REFERENCE)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, means, yerr=[err_lo, err_hi], capsize=5, color=colors)
    for bar in bars[1:]:
        bar.set_hatch("//")
        bar.set_alpha(0.55)
    ax.axhline(6.5, color="grey", linestyle="--",
               label="6.5 = symmetric break-even")
    ax.set_ylabel("mean points per hand (lower is better)")
    ax.set_title("Phase 3 full pipeline vs Phase 2 (ghosted/hatched), "
                 f"{n_deals} deals x 4 rotations")
    ax.legend()
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, png_name), dpi=150)
    plt.close(fig)


def run_choice_only(args):
    """The honest-CHOICE row alone: probe, budget check, run, ablation3.txt."""
    global ACTIVE_PARTIAL
    ACTIVE_PARTIAL = PARTIAL3

    name, kind, cid, level_name, respects = CHOICE_CONFIG
    if args.smoke:
        smoke_seeds = ra.DEAL_SEEDS[:2]
        with open(PARTIAL3, "w") as f:
            f.write(f"# choice smoke start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        t0 = time.time()
        row = run_config_serial(name, kind, cid, level_name, respects,
                                smoke_seeds)
        write_choice_outputs(
            row, len(smoke_seeds), smoke_seeds, s_per_game=0.0,
            note="SMOKE RUN: 2 deals, mechanics check only -- the mean is "
                 "not meaningful and must not be compared to the phase-2 "
                 "rows below.",
            elapsed=time.time() - t0, workers=1,
            out_name="ablation3_smoke.txt", png_name="ablation3_smoke.png")
        print("[choice-smoke] done", flush=True)
        return

    s_per_game = choice_timing_probe()
    n_deals = N_DEALS_TARGET
    note = None
    projected = 4 * n_deals * s_per_game
    print(f"[probe] projected total at {n_deals} deals = "
          f"{projected/3600:.2f}h serial "
          f"({projected/args.workers/3600:.2f}h at {args.workers} workers)",
          flush=True)
    # Budget rule is stated in wall-clock terms, so the worker count counts.
    if projected / args.workers > BUDGET_SECONDS:
        n_deals = 250
        note = (f"projected {projected/args.workers/3600:.2f}h at 500 deals "
                f"exceeded the 2h budget; shrunk to 250 deals per the compute "
                f"budget rule.")
        print(f"[probe] {note}", flush=True)
        projected = 4 * n_deals * s_per_game
        if projected / args.workers > BUDGET_SECONDS:
            print(f"\nSTOP: even 250 deals projects "
                  f"{projected/args.workers/3600:.2f}h, over the 2h budget. "
                  "Per the plan the lead must consult the owner. Nothing has "
                  "been run.", flush=True)
            return

    deal_seeds = ra.DEAL_SEEDS[:n_deals]
    with open(PARTIAL3, "w") as f:
        f.write(f"# choice run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t0 = time.time()
    results, tallies = run_all(deal_seeds, args.workers, configs=[CHOICE_CONFIG])
    elapsed = time.time() - t0
    row = as_rows(results, tallies, deal_seeds, elapsed,
                  configs=[CHOICE_CONFIG])[0]
    write_choice_outputs(row, n_deals, deal_seeds, s_per_game, note, elapsed,
                         args.workers)
    print(f"[done] wall time {elapsed/60:.1f} min with {args.workers} workers",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 deals per row, serial, then exit (mechanics check)")
    ap.add_argument("--choice-only", action="store_true",
                    help="run ONLY the Phase-3 honest-CHOICE row (config_id "
                         "310) and write results/ablation3.txt; combine with "
                         "--smoke for a 2-deal mechanics check")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if args.choice_only:
        run_choice_only(args)
        return

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
