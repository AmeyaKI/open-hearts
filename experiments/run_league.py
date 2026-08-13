"""Phase 4 Task 8: bot league -- the standing cross-version benchmark.

Round-robin: ROW bot plays 1-vs-3 against COLUMN bot's trio, on identical
deals, rotated through all 4 seats (the usual `rotated_match` contract).
6 entrants x 6 entrants = 36 cells, including the diagonal (self-mirror --
every diagonal cell should bracket 6.5, the symmetric break-even; that's
the built-in sanity alarm per row/column "family").

Entrants (names exactly as printed):
    1. random         RandomPlayer
    2. heuristic       HeuristicPlayer
    3. p1-search-FULL  Phase-1 SearchPlayer(Level.FULL, n_samples=100),
                        jit_sampler=False (default, bitwise-pinned Phase-1
                        construction; see run_ablation.py / run_ablation2.py).
    4. honest-FULL      HonestSearchPlayer(Level.FULL, n_outer=50, n_inner=20),
                        table-sampled outer worlds (posterior_factory=None,
                        the Phase-2 config; see run_ablation2.py's
                        honest_factory).
    5. honest-CHOICE    HonestSearchPlayer(Level.FULL, n_outer=50, n_inner=20),
                        OUTER worlds from WeightedPosterior(Level.FULL,
                        HeuristicPlayer(), epsilon=0, n_worlds=n_outer) --
                        the bridge construction, config_id=310, from
                        run_ablation2.py / run_ablation4.py.
    6. value-h1         ValueSearchPlayer(Level.FULL, n_outer=50, n_inner=20,
                        horizon=1), same CHOICE posterior construction as
                        honest-CHOICE -- ablation4's row "value-CHOICE-h1",
                        the best Phase-4 learned bot, included honestly as
                        (mostly) a loser.

Determinism: for the ROW seat, `bot_factory()` is called once per game (the
harness contract) and derives its rng from
`(config_id_row(row, col), deal_seed, rotation)` via `run_ablation._game_seed`
(3-tuple seeding, bitwise-identical convention to every prior ablation
script). For the COLUMN trio, `opp_factory()` is called 3x per game (once
per remaining seat, in harness seat order) -- there is no 3-arg precedent
for that in this codebase, so this script defines its own 4-tuple seeding,
`_game_seed_seat(config_id, deal_seed, rotation, seat)`, giving each of the
3 trio instances its own deterministic, reproducible rng stream. Random and
Heuristic entrants are included in this same factory machinery (even though
HeuristicPlayer draws no randomness) so every cell -- diagonal included --
is seeded uniformly per (cell, deal, rotation, seat).

`config_id_row(row, col) = 10_000 + row * 6 + col`
`config_id_col(row, col) = 20_000 + row * 6 + col`
distinct per cell AND distinct between the row-role and column-role streams,
so the same entrant playing ROW vs COLUMN in the same cell never shares an
rng stream with itself.

Budget: search-vs-search cells (rows/cols 3-6) are by far the most
expensive; --probe estimates s/game for the 4 representative cell classes
(cheap/cheap, cheap/expensive, expensive/expensive, and the value-h1 row)
plus a total projection DIVIDED BY --workers against the 2h cap, matching
run_ablation4.py's budget-projection convention (and its explicit fix of
ablation2's missing division).

Usage:
    python experiments/run_league.py --smoke    # tiny matrix, serial, exit
    python experiments/run_league.py --probe    # timing probe only, exit
    python experiments/run_league.py             # full league (LEAD ONLY)
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
from openhearts.players.random_player import RandomPlayer  # noqa: E402
from openhearts.search.decision import SearchPlayer  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402
from openhearts.search.valuesearch import ValueSearchPlayer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
PARTIAL = os.path.join(RESULTS, "league_partial.txt")
SMOKE_PARTIAL = os.path.join(RESULTS, "league_smoke_partial.txt")

N_DEALS_TARGET = 100
BUDGET_SECONDS = 2 * 3600
WORKERS = 8
CHUNK = 25
MEM_LIMIT_GB = 100.0

BRIDGE_N_OUTER = 50
BRIDGE_N_INNER = 20
P1_N_SAMPLES = 100

CHOICE_EPSILON = 0.0
CHOICE_MAX_DRAWS = 50_000

ENTRANTS = ["random", "heuristic", "p1-search-FULL", "honest-FULL",
            "honest-CHOICE", "value-h1"]

# Known pairwise results (all "vs 3 plain HeuristicPlayer opponents"), for
# the sanity comparison printed against this league's heuristic COLUMN.
KNOWN_VS_HEURISTIC = {
    "heuristic": 6.5,          # mirror -- exact, not measured
    "p1-search-FULL": 3.27,    # run_ablation.py FULL row
    "honest-FULL": 3.25,       # run_ablation2.py honest-FULL row
    "honest-CHOICE": 2.87,     # run_ablation2.py / run_ablation4.py bridge
    "value-h1": None,          # not independently published pre-league
}


def _game_seed_seat(config_id, deal_seed, rotation, seat):
    """4-tuple seeding for the 3 column-trio instances of one game -- each
    gets its own deterministic, reproducible rng stream, distinct from the
    others and from the row bot's stream."""
    return np.random.default_rng([config_id, deal_seed, rotation, seat])


def _posterior_factory(n_worlds):
    """CHOICE posterior: WeightedPosterior at eps=0 against HeuristicPlayer,
    identical construction to run_ablation2.py's honest-CHOICE row and
    run_ablation4.py's bridge row, tied to this bot's own n_outer."""
    def factory(view, rng):
        return WeightedPosterior.from_view(
            view, Level.FULL, HeuristicPlayer(), CHOICE_EPSILON,
            n_worlds, rng=rng, max_draws=CHOICE_MAX_DRAWS, keep_worlds=True)
    return factory


def build_bot(entrant, rng, tally=None):
    """Construct one entrant's bot instance from its own rng stream."""
    if entrant == "random":
        bot = RandomPlayer(rng)
    elif entrant == "heuristic":
        bot = HeuristicPlayer()
    elif entrant == "p1-search-FULL":
        bot = SearchPlayer(Level.FULL, P1_N_SAMPLES, rng)
    elif entrant == "honest-FULL":
        bot = HonestSearchPlayer(Level.FULL, BRIDGE_N_OUTER, BRIDGE_N_INNER,
                                 rng, sampler_respects_voids=True)
    elif entrant == "honest-CHOICE":
        bot = HonestSearchPlayer(Level.FULL, BRIDGE_N_OUTER, BRIDGE_N_INNER,
                                 rng, sampler_respects_voids=True,
                                 posterior_factory=_posterior_factory(BRIDGE_N_OUTER))
    elif entrant == "value-h1":
        bot = ValueSearchPlayer(Level.FULL, BRIDGE_N_OUTER, BRIDGE_N_INNER,
                                rng, sampler_respects_voids=True,
                                posterior_factory=_posterior_factory(BRIDGE_N_OUTER),
                                horizon=1)
    else:
        raise ValueError(entrant)
    if tally is not None:
        tally.setdefault("bots", []).append(bot)
    return bot


def _row_factory(entrant, config_id, deal_seeds, tally=None):
    """1 call/game (harness contract for `bot_factory`)."""
    counter = itertools.count()
    def factory():
        i = next(counter)
        seed, rotation = deal_seeds[i // 4], i % 4
        rng = ra._game_seed(config_id, seed, rotation)
        return build_bot(entrant, rng, tally)
    return factory


def _col_factory(entrant, config_id, deal_seeds, tally=None):
    """3 calls/game (harness contract for `opp_factory`, one per remaining
    seat, in seat order) -- each of the 3 trio instances gets its own seat
    slot in the 4-tuple rng seed."""
    counter = itertools.count()
    def factory():
        i = next(counter)
        game_i, seat = i // 3, i % 3
        seed, rotation = deal_seeds[game_i // 4], game_i % 4
        rng = _game_seed_seat(config_id, seed, rotation, seat)
        return build_bot(entrant, rng, tally)
    return factory


def _counters(entrant, tally):
    """Sum (fallbacks, failed_samples, inner_fallbacks, inner_failed_samples,
    posterior [collapses, decisions_served, worlds_supplied])."""
    bots = tally.get("bots", []) if tally is not None else []
    if entrant in ("random", "heuristic"):
        return 0, 0, None, None, None
    fb = sum(b.fallbacks for b in bots)
    fs = sum(b.failed_samples for b in bots)
    if entrant == "p1-search-FULL":
        return fb, fs, None, None, None
    ifb = sum(b.inner_fallbacks for b in bots)
    ifs = sum(b.inner_failed_samples for b in bots)
    pc = [sum(b.posterior_collapses for b in bots),
          sum(b.posterior_decisions for b in bots),
          sum(b.posterior_worlds for b in bots)]
    return fb, fs, ifb, ifs, pc


def cell_name(row, col):
    return f"{row}__vs__{col}"


def config_id_row(row, col):
    return 10_000 + row * 6 + col


def config_id_col(row, col):
    return 20_000 + row * 6 + col


# ---------------------------------------------------------------- serial
def run_cell_serial(row, col, deal_seeds):
    tally_row, tally_col = {}, {}
    bot_factory = _row_factory(row, config_id_row(ENTRANTS.index(row),
                                                   ENTRANTS.index(col)),
                               deal_seeds, tally_row)
    opp_factory = _col_factory(col, config_id_col(ENTRANTS.index(row),
                                                   ENTRANTS.index(col)),
                               deal_seeds, tally_col)
    t0 = time.time()
    per_deal = rotated_match(deal_seeds, bot_factory, opp_factory)
    elapsed = time.time() - t0
    mean, lo, hi = bootstrap_ci(per_deal)
    rtally = _counters(row, tally_row)
    ctally = _counters(col, tally_col)
    return {"row": row, "col": col, "per_deal": per_deal, "mean": mean,
            "lo": lo, "hi": hi, "seconds": elapsed,
            "rtally": rtally, "ctally": ctally}


# ---------------------------------------------------------------- parallel
def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield i, seq[i:i + size]


def worker(row, col, start, chunk_seeds):
    tally_row, tally_col = {}, {}
    r_idx, c_idx = ENTRANTS.index(row), ENTRANTS.index(col)
    bot_factory = _row_factory(row, config_id_row(r_idx, c_idx),
                               chunk_seeds, tally_row)
    opp_factory = _col_factory(col, config_id_col(r_idx, c_idx),
                               chunk_seeds, tally_col)
    per_deal = rotated_match(chunk_seeds, bot_factory, opp_factory)
    rfb, rfs, rifb, rifs, rpc = _counters(row, tally_row)
    cfb, cfs, cifb, cifs, cpc = _counters(col, tally_col)
    return (row, col, start, per_deal,
            (rfb, rfs, rifb, rifs, rpc), (cfb, cfs, cifb, cifs, cpc))


def total_rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


def _append_partial(name, values, path=PARTIAL):
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "a") as f:
        f.write(name + " " + " ".join(f"{v:.6f}" for v in values) + "\n")
        f.flush()


def _load_partial(deal_seeds):
    """Parse an existing PARTIAL file into {(row, col, start): per_deal},
    so a resumed run can skip chunks already completed. Lines look like
    '<row>__vs__<col>@<start> v1 v2 v3 ...'; comment/malformed lines ignored."""
    done = {}
    if not os.path.exists(PARTIAL):
        return done
    with open(PARTIAL) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                head, *vals = line.split()
                name, start = head.rsplit("@", 1)
                row, col = name.split("__vs__")
                start = int(start)
                per_deal = np.array([float(v) for v in vals])
            except Exception:
                continue
            done[(row, col, start)] = per_deal
    return done


def run_all(deal_seeds, workers, pairs, resume=True):
    import concurrent.futures as cf
    results = {(r, c): np.zeros(len(deal_seeds)) for r, c in pairs}
    cell_tallies = {(r, c): (None, None) for r, c in pairs}
    already = _load_partial(deal_seeds) if resume else {}
    jobs_meta = []
    n_resumed = 0
    for row, col in pairs:
        for start, chunk in _chunks(deal_seeds, CHUNK):
            key = (row, col, start)
            if key in already and len(already[key]) == len(chunk):
                results[(row, col)][start:start + len(chunk)] = already[key]
                n_resumed += 1
                print(f"[resume] {row} vs {col} deals {start}-"
                      f"{start + len(chunk) - 1} already in {PARTIAL}, "
                      "skipping", flush=True)
                continue
            jobs_meta.append((row, col, start, chunk))
    if n_resumed:
        print(f"[resume] {n_resumed} chunk(s) skipped (already in "
              f"{PARTIAL}) -- their fallback/posterior counters are NOT "
              "included in this invocation's totals (only per-deal scores "
              "are recovered).", flush=True)
    if not jobs_meta:
        print("[resume] all chunks already complete", flush=True)
        return results, cell_tallies, n_resumed
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(worker, row, col, start, chunk)
                for row, col, start, chunk in jobs_meta]
        done = 0
        t0 = time.time()
        for fut in cf.as_completed(jobs):
            row, col, start, per_deal, rtally, ctally = fut.result()
            results[(row, col)][start:start + len(per_deal)] = per_deal
            prev_r, prev_c = cell_tallies[(row, col)]
            cell_tallies[(row, col)] = (_merge_tally(prev_r, rtally),
                                        _merge_tally(prev_c, ctally))
            done += 1
            mem = total_rss_gb()
            print(f"[{done}/{len(jobs)}] {row} vs {col} deals {start}-"
                  f"{start + len(per_deal) - 1} done | mem={mem:.1f}GB | "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)
            _append_partial(f"{cell_name(row, col)}@{start}", per_deal)
            if mem > MEM_LIMIT_GB:
                print(f"ABORT: memory {mem:.1f}GB exceeds "
                      f"{MEM_LIMIT_GB}GB limit", flush=True)
                for j in jobs:
                    j.cancel()
                raise MemoryError("memory limit exceeded")
    return results, cell_tallies, n_resumed


def _merge_tally(prev, new):
    if new is None:
        return prev
    fb, fs, ifb, ifs, pc = new
    if prev is None:
        return (fb, fs, ifb, ifs, list(pc) if pc is not None else None)
    pfb, pfs, pifb, pifs, ppc = prev
    mfb, mfs = pfb + fb, pfs + fs
    mifb = None if ifb is None else (pifb or 0) + ifb
    mifs = None if ifs is None else (pifs or 0) + ifs
    mpc = None if pc is None else [ (ppc[k] if ppc else 0) + pc[k] for k in range(3) ]
    return (mfb, mfs, mifb, mifs, mpc)


# ---------------------------------------------------------------- probe
# Representative cell classes for the probe: cheap/cheap, cheap/expensive
# (row cheap, column trio expensive -- 3x the search cost), expensive/cheap,
# expensive/expensive (the priciest class), plus the value-h1 row explicitly
# (it carries value-net inference on top of search).
PROBE_CELLS = [
    ("random", "heuristic", "cheap/cheap"),
    ("heuristic", "honest-CHOICE", "cheap-row/expensive-col"),
    ("honest-CHOICE", "heuristic", "expensive-row/cheap-col"),
    ("honest-CHOICE", "honest-CHOICE", "expensive/expensive"),
    ("value-h1", "honest-CHOICE", "value-h1-row/expensive-col"),
    # CHOICE posterior's eps=0 filtering is only exact when the opponents ARE
    # HeuristicPlayer (see _posterior_factory / ablation4's docstring). vs
    # random/search/value trios, observed plays are inconsistent with that
    # model -- worlds get filtered out, the posterior can collapse or burn
    # toward max_draws per decision. Priced as its own class and reported,
    # though the additive model below does not lean on it for projection.
    ("honest-CHOICE", "random", "expensive-row/random-col"),
    # Every non-cheap entrant gets BOTH a rowcost probe (X vs heuristic) and
    # a colcost probe (heuristic vs X): p1-search-FULL (Python reference
    # sampler, jit_sampler=False) and honest-FULL (table-sampled, no
    # WeightedPosterior) are DIFFERENT constructions from honest-CHOICE and
    # are not assumed to share its s/game. This gives an additive cost
    # model, cost(R, C) ~= rowcost(R) + colcost(C), rather than lumping the
    # 12 search-vs-search cells -- which dominate the budget -- onto the
    # single honest-CHOICE-vs-honest-CHOICE probe cell.
    ("p1-search-FULL", "heuristic", "p1-search-FULL row"),
    ("heuristic", "p1-search-FULL", "p1-search-FULL col"),
    ("honest-FULL", "heuristic", "honest-FULL row"),
    ("heuristic", "honest-FULL", "honest-FULL col"),
    ("value-h1", "heuristic", "value-h1 row"),
    ("heuristic", "value-h1", "value-h1 col"),
]

# honest-CHOICE's rowcost/colcost come from cells already in PROBE_CELLS
# under different class labels (reused rather than re-probed).
_ROWCOST_CLASS = {
    "p1-search-FULL": "p1-search-FULL row",
    "honest-FULL": "honest-FULL row",
    "honest-CHOICE": "expensive-row/cheap-col",
    "value-h1": "value-h1 row",
}
_COLCOST_CLASS = {
    "p1-search-FULL": "p1-search-FULL col",
    "honest-FULL": "honest-FULL col",
    "honest-CHOICE": "cheap-row/expensive-col",
    "value-h1": "value-h1 col",
}
_CHEAP = {"random", "heuristic"}


def timing_probe(n_probe_deals=2):
    probe_seeds = [900_000 + i for i in range(n_probe_deals)]
    s_per_game = {}
    print(f"[probe] {n_probe_deals} deals x 4 rotations per probe cell",
          flush=True)
    for row, col, cls in PROBE_CELLS:
        cell = run_cell_serial(row, col, probe_seeds)
        n_games = 4 * n_probe_deals
        spg = cell["seconds"] / n_games
        s_per_game[(row, col)] = (spg, cls)
        print(f"[probe] {row} vs {col} [{cls}]: {spg:.3f}s/game "
              f"({cell['seconds']:.1f}s for {n_games} games)", flush=True)
    return s_per_game, probe_seeds


def project_total_seconds(s_per_game, n_deals):
    """Projects the FULL 36-cell matrix additively: cost(R, C) ~=
    rowcost(R) + colcost(C), where rowcost(R) is R's probed s/game as the
    ROW bot against a cheap (heuristic) trio, and colcost(C) is a cheap
    (heuristic) row bot's probed s/game against C's trio. Cheap entrants
    (random, heuristic) contribute ~0 either way (measured s/game ~0.000),
    so this reduces to the single honest-CHOICE-vs-honest-CHOICE-style
    lumping ONLY for the cheap/cheap cell; every other cell gets its own
    rowcost+colcost pricing instead of being bucketed into one class.
    `honest-CHOICE vs random` is probed and reported separately as a
    sanity check on the additive approximation (posterior collapse against
    non-heuristic trios can behave differently than vs-heuristic pricing
    predicts) but is not used for the projection itself.
    """
    games = 4 * n_deals
    rowcost, colcost = _costs(s_per_game)
    total = 0.0
    per_cell_seconds = {}
    for row in ENTRANTS:
        for col in ENTRANTS:
            spg = rowcost[row] + colcost[col]
            per_cell_seconds[(row, col)] = spg * games
            total += spg * games
    return total, per_cell_seconds


def _costs(s_per_game):
    by_class = {cls: spg for (_r, _c), (spg, cls) in s_per_game.items()}
    rowcost = {e: (by_class[_ROWCOST_CLASS[e]] if e not in _CHEAP else 0.0)
              for e in ENTRANTS}
    colcost = {e: (by_class[_COLCOST_CLASS[e]] if e not in _CHEAP else 0.0)
              for e in ENTRANTS}
    return rowcost, colcost


def additive_model_check(s_per_game):
    """Prints predicted (rowcost+colcost) vs measured s/game for every
    PROBED cell -- the held-out validation for project_total_seconds'
    additive approximation. Cells where row or col is cheap are trivially
    exact (both terms are ~0 or the identity); the informative residuals
    are the search-vs-search / search-vs-random cells that were probed but
    are NOT among the 12 cells the model was fit from
    (honest-CHOICE-vs-honest-CHOICE, value-h1-vs-honest-CHOICE,
    honest-CHOICE-vs-random)."""
    rowcost, colcost = _costs(s_per_game)
    print("[probe] additive model check (predicted = rowcost[row] + "
          "colcost[col] vs measured):", flush=True)
    for (row, col), (spg, cls) in s_per_game.items():
        pred = rowcost[row] + colcost[col]
        pct = 100 * (pred / spg - 1) if spg > 0 else 0.0
        print(f"[probe]   {row} vs {col} [{cls}]: predicted={pred:.3f} "
              f"measured={spg:.3f} ({pct:+.0f}%)", flush=True)


def print_budget_projection(s_per_game, n_deals, workers):
    total, per_cell = project_total_seconds(s_per_game, n_deals)
    projected_wall = total / workers
    print("=" * 72, flush=True)
    print(f"[BUDGET] projected total worker-seconds for the full 36-cell "
          f"matrix at {n_deals} deals/cell = {total:.0f}s "
          f"({total/3600:.2f}h serial)", flush=True)
    print(f"[BUDGET] DIVIDED BY {workers} workers -> wall clock = "
          f"{projected_wall:.0f}s ({projected_wall/3600:.2f}h) "
          f"vs {BUDGET_SECONDS/3600:.1f}h cap", flush=True)
    print("=" * 72, flush=True)
    return total, projected_wall


# ---------------------------------------------------------------- outputs
def write_outputs(matrix, n_deals, deal_seeds, s_per_game, note, elapsed,
                  workers, entrants=None, cell_tallies=None, n_resumed=0,
                  out_name="league.txt", png_name="league.png"):
    os.makedirs(RESULTS, exist_ok=True)
    entrants = entrants if entrants is not None else ENTRANTS
    by_class = {cls: spg for (spg, cls) in s_per_game.values()}
    lines = [
        "open-hearts bot league (Phase 4 Task 8) -- round robin, ROW bot "
        "1-vs-3 against COLUMN bot's trio.",
        "Lower points per hand is better for the ROW bot. 26 points are "
        "dealt out per hand, so 6.5 is the symmetric break-even.",
        "",
        f"deals: {n_deals} (seeds {deal_seeds[0]}..{deal_seeds[-1]}), each "
        "played 4x with the ROW bot rotated through every seat; identical "
        "deals for every cell. CIs are 10,000 bootstrap resamples over "
        "DEALS (not games).",
        "",
        "entrants:",
        "  1. random           RandomPlayer",
        "  2. heuristic        HeuristicPlayer",
        "  3. p1-search-FULL   Phase-1 SearchPlayer(Level.FULL, "
        f"n_samples={P1_N_SAMPLES}), jit_sampler=False",
        "  4. honest-FULL      HonestSearchPlayer(Level.FULL, "
        f"n_outer={BRIDGE_N_OUTER}, n_inner={BRIDGE_N_INNER}), table-sampled "
        "outer worlds (Phase-2 config)",
        "  5. honest-CHOICE    HonestSearchPlayer(Level.FULL, "
        f"n_outer={BRIDGE_N_OUTER}, n_inner={BRIDGE_N_INNER}), CHOICE "
        "posterior (WeightedPosterior, Level.FULL proposal, HeuristicPlayer, "
        f"epsilon={CHOICE_EPSILON}, n_worlds={BRIDGE_N_OUTER}, "
        f"max_draws={CHOICE_MAX_DRAWS}) -- bridge construction "
        "(config_id=310 in run_ablation2.py/run_ablation4.py)",
        "  6. value-h1         ValueSearchPlayer(Level.FULL, "
        f"n_outer={BRIDGE_N_OUTER}, n_inner={BRIDGE_N_INNER}, horizon=1), "
        "same CHOICE posterior construction",
        "",
        "rng: ROW bot seeded from (config_id_row(row,col), deal_seed, "
        "rotation) via run_ablation._game_seed (1 call/game, harness "
        "contract). COLUMN trio seeded from (config_id_col(row,col), "
        "deal_seed, rotation, seat) -- 3 calls/game, one per trio seat, "
        "distinct rng per instance. config_id_row/col are distinct per "
        "cell AND between row/column roles.",
        f"timing probe s/game per class: " +
        ", ".join(f"{cls}={spg:.3f}" for cls, spg in by_class.items()),
        f"wall time: {elapsed:.1f}s with {workers} worker(s)" if elapsed else "",
    ]
    if note:
        lines += ["", f"NOTE: {note}"]

    lines += ["", "SANITY: diagonal cells (self-mirror) must bracket 6.5."]
    for e in entrants:
        r, lo, hi = matrix[(e, e)]
        ok = lo <= 6.5 <= hi
        lines.append(f"  {e:<18} diagonal mean={r:.3f} CI=({lo:.3f},{hi:.3f}) "
                     f"{'OK' if ok else '*** DOES NOT BRACKET 6.5 ***'}")

    if "heuristic" in entrants:
        lines += ["", "SANITY: known pairwise results vs the heuristic "
                  "column (row bot vs 3 plain HeuristicPlayer opponents):"]
        for e in entrants:
            known = KNOWN_VS_HEURISTIC.get(e)
            r, lo, hi = matrix[(e, "heuristic")]
            known_str = f"{known:.2f}" if known is not None else "n/a"
            lines.append(f"  {e:<18} league mean={r:.3f} CI=({lo:.3f},{hi:.3f}) "
                         f"known={known_str}")

    if cell_tallies:
        resumed_caveat = (
            f" NOTE: {n_resumed} chunk(s) in this run were resumed from "
            f"{PARTIAL} -- their fallback/posterior counters are NOT "
            "included below (only per-deal scores were recovered), so "
            "these counts under-report the full run." if n_resumed else "")
        lines += ["", "posterior collapse rate (ROW-role, COL-role) per "
                  "cell -- decisions where NO candidate world survived "
                  "CHOICE filtering and the bot fell back to the "
                  "constraint sampler; n/a for entrants without a "
                  "posterior (random/heuristic/p1-search-FULL)." +
                  resumed_caveat]
        for row in entrants:
            for col in entrants:
                rt, ct = cell_tallies.get((row, col), (None, None))
                def _rate(t):
                    if t is None or t[4] is None or t[4][1] == 0:
                        return "n/a"
                    return f"{t[4][0]}/{t[4][1]}"
                lines.append(f"  {row:<18} vs {col:<18} "
                             f"row={_rate(rt)} col={_rate(ct)}")

    lines += ["", "matrix (ROW bot mean pts/hand vs COLUMN bot's trio; "
              "[lo95,hi95] below each cell):",
              " " * 18 + "".join(f"{c:>18}" for c in entrants)]
    for row in entrants:
        mean_line = f"{row:<18}"
        ci_line = " " * 18
        for col in entrants:
            m, lo, hi = matrix[(row, col)]
            mean_line += f"{m:>18.3f}"
            ci_line += f"[{lo:>6.3f},{hi:>6.3f}]".rjust(18)
        lines.append(mean_line)
        lines.append(ci_line)

    with open(os.path.join(RESULTS, out_name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # heatmap, diverging around 6.5
    grid = np.array([[matrix[(r, c)][0] for c in entrants] for r in entrants])
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = max(abs(grid.min() - 6.5), abs(grid.max() - 6.5), 1.0)
    im = ax.imshow(grid, cmap="RdBu", vmin=6.5 - vmax, vmax=6.5 + vmax)
    ax.set_xticks(range(len(entrants)))
    ax.set_yticks(range(len(entrants)))
    ax.set_xticklabels(entrants, rotation=35, ha="right")
    ax.set_yticklabels(entrants)
    ax.set_xlabel("COLUMN bot (trio, 3 seats)")
    ax.set_ylabel("ROW bot (1 seat)")
    ax.set_title(f"Bot league: ROW pts/hand vs COLUMN trio\n"
                 f"{n_deals} deals x 4 rotations (6.5 = break-even)")
    for i in range(len(entrants)):
        for j in range(len(entrants)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="mean pts/hand (row bot)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, png_name), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny matrix (3 entrants), 2 deals/cell, serial, "
                         "then exit (mechanics check)")
    ap.add_argument("--probe", action="store_true",
                    help="timing probe over representative cell classes, "
                         "then exit")
    ap.add_argument("--deals", type=int, default=N_DEALS_TARGET)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if args.smoke:
        # Uses its own checkpoint file, NOT `PARTIAL` -- `PARTIAL` is the
        # full run's resumable checkpoint; a --smoke invocation must never
        # truncate or interleave with it (see SMOKE_PARTIAL).
        smoke_entrants = ["random", "heuristic", "honest-CHOICE"]
        smoke_seeds = ra.DEAL_SEEDS[:2]
        with open(SMOKE_PARTIAL, "w") as f:
            f.write(f"# smoke run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        t0 = time.time()
        matrix = {}
        cell_tallies = {}
        for row in smoke_entrants:
            for col in smoke_entrants:
                cell = run_cell_serial(row, col, smoke_seeds)
                matrix[(row, col)] = (cell["mean"], cell["lo"], cell["hi"])
                cell_tallies[(row, col)] = (cell["rtally"], cell["ctally"])
                _append_partial(cell_name(row, col), cell["per_deal"],
                               path=SMOKE_PARTIAL)
                print(f"[smoke] {row} vs {col}: mean={cell['mean']:.3f}",
                      flush=True)
        elapsed = time.time() - t0
        s_per_game = {(row, col): (0.0, "smoke")
                      for row in smoke_entrants for col in smoke_entrants}
        write_outputs(matrix, len(smoke_seeds), smoke_seeds, s_per_game,
                     note="SMOKE RUN: 3 entrants, 2 deals/cell, mechanics "
                          "check only -- means are not meaningful.",
                     elapsed=elapsed, workers=1, entrants=smoke_entrants,
                     cell_tallies=cell_tallies,
                     out_name="league_smoke.txt", png_name="league_smoke.png")
        print(f"[smoke] done in {elapsed:.1f}s", flush=True)
        return

    if args.probe:
        s_per_game, probe_seeds = timing_probe()
        additive_model_check(s_per_game)
        print_budget_projection(s_per_game, args.deals, args.workers)
        print("[probe] done (no full run performed)", flush=True)
        return

    s_per_game, probe_seeds = timing_probe()
    additive_model_check(s_per_game)
    n_deals = args.deals
    note = None
    total, projected_wall = print_budget_projection(s_per_game, n_deals,
                                                     args.workers)
    if projected_wall > BUDGET_SECONDS:
        # Shrink proportionally to the largest deal count that fits the
        # budget (floored to a CHUNK multiple), not a fixed cliff -- being
        # 2% over the cap should not quarter the deal count and quadruple
        # every CI width.
        n_fit = int(n_deals * BUDGET_SECONDS / projected_wall)
        n_fit = max(CHUNK, (n_fit // CHUNK) * CHUNK)
        n_deals = n_fit
        total, projected_wall = print_budget_projection(s_per_game, n_deals,
                                                         args.workers)
        note = (f"projected {projected_wall/3600:.2f}h at {args.deals} deals "
                f"exceeded the 2h budget; shrunk to {n_deals} deals (largest "
                "CHUNK-aligned count that fits) per the compute budget rule "
                "(shrink deals before dropping entrants).")
        print(f"[probe] {note}", flush=True)
        if projected_wall > BUDGET_SECONDS:
            print(f"\nSTOP: even {n_deals} deals projects "
                  f"{projected_wall/3600:.2f}h, over the 2h budget. Per the "
                  "plan the lead must consult the owner. Nothing has been "
                  "run.", flush=True)
            return

    deal_seeds = ra.DEAL_SEEDS[:n_deals]
    pairs = [(r, c) for r in ENTRANTS for c in ENTRANTS]
    # PARTIAL is append-only and resumable: do NOT truncate an existing file
    # (a prior interrupted run's chunks are read back and skipped by
    # run_all's resume logic). Only create it fresh if it doesn't exist yet.
    if not os.path.exists(PARTIAL):
        with open(PARTIAL, "w") as f:
            f.write(f"# run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    else:
        with open(PARTIAL, "a") as f:
            f.write(f"# resumed {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t0 = time.time()
    results, cell_tallies, n_resumed = run_all(deal_seeds, args.workers, pairs)
    elapsed = time.time() - t0
    matrix = {}
    for (row, col), per_deal in results.items():
        mean, lo, hi = bootstrap_ci(per_deal)
        matrix[(row, col)] = (mean, lo, hi)
    write_outputs(matrix, n_deals, deal_seeds, s_per_game, note, elapsed,
                 args.workers, entrants=ENTRANTS, cell_tallies=cell_tallies,
                 n_resumed=n_resumed)
    print(f"[done] wall time {elapsed/60:.1f} min with {args.workers} workers",
          flush=True)


if __name__ == "__main__":
    main()
