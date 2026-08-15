"""Phase 5 Task 6: THE HEADLINE ABLATION.

Eight rows vs 20 HELD-OUT personality trios (500 deals, seeds 100000..100499,
partitioned into 20 consecutive MATCH-BLOCKS of 25 deals; each block plays
against one fixed trio -- deal index -> block index is a pure function,
identical across every row, so the paired per-deal bootstrap is unaffected):

    1. honest-FULL           constraint-only outer worlds (the robust floor).
    2. honest-CHOICE-strict  heuristic-policy posterior, eps=0 (expected to
                              collapse against non-heuristic opponents).
    3. honest-CHOICE-soft    heuristic-policy posterior, eps=0.1 -- THE
                              BASELINE TO BEAT.
    4. profiled-R            Organ 1 only: profiler reading (GENERIC),
                              heuristic playouts.
    5. profiled-RI           Organs 1+2: profiler reading + model-driven
                              (GENERIC) playouts.
    6. profiled-RIA          RI + Task-5 per-seat mixture adaptation on the
                              READING likelihood only (playouts stay GENERIC,
                              matching search/profiled.py's row table).
    7. profiled-ORACLE       reading with the CONDITIONED net fed the TRUE
                              per-block trio parameters (simulation-only
                              ceiling) + GENERIC playouts. FALLBACK, stated
                              prominently: kernel_profiled's playout kernel
                              takes one GENERIC 6-tuple with no per-seat
                              parameter block, so the plan's "oracle reading +
                              oracle playouts" variant is not supported
                              in-kernel; this row is "oracle READING +
                              GENERIC playouts", per the plan's own named
                              fallback.
    8. personality-mirror    alarm row: all 4 seats are the block's first
                              trio member (population trio symmetry, should
                              bracket 6.5 pooled across blocks).

RIA MIXTURE PROTOCOL (lead decision, implemented here verbatim). Each block
keeps THREE persistent `SeatMixture` objects, one per TRIO POSITION (0,1,2)
-- not per absolute seat. Before every game, the harness computes the
ground-truth mapping {absolute seat -> trio position} from that game's
rotation (the bot occupies seat = rotation; the other three seats, in
ascending order, are trio positions 0,1,2) and points the
`AdaptedLikelihood.mixtures` dict at the right persistent objects for that
seating. After the game, `observe_hand(history)` is called, which updates
whichever mixture objects are currently referenced by the (still-correct)
seat mapping. Mixtures reset only at block boundaries (a fresh bot, and
fresh `SeatMixture`s, are constructed per block). The identical mechanism
serves `profiled-ORACLE`'s seat_params (trio position -> true param vector).

OPPONENT SEEDING. Rows 1-7 share ONE opponent config id (`OPP_CONFIG_ID`), so
every row sees the identical personality behaviour stream at
(deal, rotation, seat) -- only the bot differs. `personality-mirror` uses its
own id (it fills all 4 seats, not 3). Bot rng is seeded once per BLOCK from
`(row_config_id, block_idx)` and advances across the block's 100 games (25
deals x 4 rotations); this is a new convention for a new script, documented
rather than borrowed, because the RIA row structurally requires ONE
persistent bot per block and the other 7 rows are kept on the identical
scheme for uniformity.

MECHANICS. Chunked by BLOCK (a worker owns one whole block for one row --
mixture state cannot cross workers). Checkpointed partials
(`results/ablation5_partial.txt`, resumable). `--probe` (2 blocks' first 3
deals, all rows, s/game + budget projection DIVIDED BY WORKERS). `--smoke`
(1 block, 3 deals, serial, all rows -- mechanics check).

Usage:
    python experiments/run_ablation5.py --smoke
    python experiments/run_ablation5.py --probe
    python experiments/run_ablation5.py            # full run, LEAD ONLY
"""
import argparse
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openhearts.belief.table import Level  # noqa: E402
from openhearts.belief.weighted import WeightedPosterior  # noqa: E402
from openhearts.engine.game import deal, play_game  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.opponent.adapt import (AdaptedLikelihood, PoolProfiler,  # noqa: E402
                                       pool_ids, train_heldout_ids)
from openhearts.opponent.infer import load_profiler  # noqa: E402
from openhearts.opponent.params import PARAM_DIM, param_vector  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.players.personality import PersonalityPlayer, sample_personality  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402
from openhearts.search.profiled import (ProfiledSearchPlayer, ProfilerLikelihood,  # noqa: E402
                                        profiler_posterior_factory)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
PARTIAL = os.path.join(RESULTS, "ablation5_partial.txt")

GENERIC_PATH = os.path.join(os.path.dirname(__file__), "..", "models",
                            "profiler_v1.npz")
CONDITIONED_PATH = os.path.join(os.path.dirname(__file__), "..", "results",
                                "profiler_train", "profiler_conditioned.npz")

N_DEALS_TARGET = 500
BLOCK_SIZE = 25
N_BLOCKS = N_DEALS_TARGET // BLOCK_SIZE
DEAL_SEED_BASE = 100_000
BUDGET_SECONDS = 2 * 3600
WORKERS = 8
MEM_LIMIT_GB = 100.0

N_OUTER = 50
N_INNER = 20
N_WORLDS = 50
CHOICE_MAX_DRAWS = 50_000
CHOICE_STRICT_EPS = 0.0
CHOICE_SOFT_EPS = 0.1

TRIO_SEED = [314159, 6060]
OPP_CONFIG_ID = 9000        # shared across rows 1-7
MIRROR_OPP_CONFIG_ID = 9001

ROW_NAMES = ["honest-FULL", "honest-CHOICE-strict", "honest-CHOICE-soft",
            "profiled-R", "profiled-RI", "profiled-RIA", "profiled-ORACLE",
            "personality-mirror"]
ROW_CONFIG_ID = {name: 5100 + i for i, name in enumerate(ROW_NAMES)}

BREAK_EVEN = 6.5
GATE_ORACLE_THRESHOLD = 0.3  # pts/hand; see PHASE5_PLAN's two-sided RL gate


# --------------------------------------------------------------- trio table
def block_trios(n_blocks=N_BLOCKS):
    """20 distinct held-out trios, deterministic (plan's rng([314159,6060]))."""
    _, held = train_heldout_ids()
    rng = np.random.default_rng(TRIO_SEED)
    trios, seen = [], set()
    while len(trios) < n_blocks:
        idx = rng.choice(len(held), size=3, replace=False)
        trio = tuple(int(held[int(i)]) for i in idx)
        key = frozenset(trio)
        if key in seen:
            continue
        seen.add(key)
        trios.append(trio)
    return trios


def quartile(i):
    """Position-in-block (0-indexed deal within a 25-deal block) -> 0..3."""
    return min(3, i // 6)


def personality_player(pid, rng):
    return PersonalityPlayer(rng, sample_personality(pid))


def opp_seat_rng(config_id, deal_seed, rotation, seat):
    return np.random.default_rng([config_id, deal_seed, rotation, seat])


def block_rng(row_name, block_idx):
    return np.random.default_rng([ROW_CONFIG_ID[row_name], block_idx])


# --------------------------------------------------------------- bot build
def choice_posterior_factory(epsilon):
    def factory(view, rng):
        return WeightedPosterior.from_view(
            view, Level.FULL, HeuristicPlayer(), epsilon, N_WORLDS, rng=rng,
            max_draws=CHOICE_MAX_DRAWS, keep_worlds=True)
    return factory


def build_bot(row_name, trio_ids, rng):
    """-> (bot, per_game_hook(seat_to_pos), extra) for a fresh block.

    `per_game_hook` re-points the likelihood's seat-keyed state (mixtures
    or seat_params) at the right persistent trio-position object before
    each game; a no-op for rows with no such state. `extra` carries the
    persistent per-trio-position objects for diagnostics/counters.
    """
    if row_name == "honest-FULL":
        bot = HonestSearchPlayer(Level.FULL, N_OUTER, N_INNER, rng,
                                 sampler_respects_voids=True,
                                 posterior_factory=None)
        return bot, (lambda s2p: None), {}

    if row_name in ("honest-CHOICE-strict", "honest-CHOICE-soft"):
        eps = CHOICE_STRICT_EPS if row_name == "honest-CHOICE-strict" else CHOICE_SOFT_EPS
        pf = choice_posterior_factory(eps)
        bot = HonestSearchPlayer(Level.FULL, N_OUTER, N_INNER, rng,
                                 sampler_respects_voids=True,
                                 posterior_factory=pf)
        return bot, (lambda s2p: None), {}

    generic_weights, _ = load_profiler(GENERIC_PATH)

    if row_name in ("profiled-R", "profiled-RI"):
        lik = ProfilerLikelihood(generic_weights)
        pf = profiler_posterior_factory(lik, level=Level.FULL,
                                        n_worlds=N_WORLDS,
                                        max_draws=CHOICE_MAX_DRAWS,
                                        keep_worlds=True)
        pw = None if row_name == "profiled-R" else generic_weights
        bot = ProfiledSearchPlayer(Level.FULL, N_OUTER, N_INNER, rng,
                                   sampler_respects_voids=True,
                                   posterior_factory=pf, playout_weights=pw)
        return bot, (lambda s2p: None), {}

    if row_name == "profiled-RIA":
        conditioned_weights, _ = load_profiler(CONDITIONED_PATH)
        pool = PoolProfiler(conditioned_weights, ids=pool_ids())
        generic_lik = ProfilerLikelihood(generic_weights)
        lik = AdaptedLikelihood(pool, mixtures={}, generic=generic_lik)
        from openhearts.opponent.adapt import SeatMixture
        identity_mixtures = {p: SeatMixture(pool) for p in range(3)}

        def hook(seat_to_pos):
            lik.mixtures = {s: identity_mixtures[p] for s, p in seat_to_pos.items()}

        pf = profiler_posterior_factory(lik, level=Level.FULL,
                                        n_worlds=N_WORLDS,
                                        max_draws=CHOICE_MAX_DRAWS,
                                        keep_worlds=True)
        bot = ProfiledSearchPlayer(Level.FULL, N_OUTER, N_INNER, rng,
                                   sampler_respects_voids=True,
                                   posterior_factory=pf,
                                   playout_weights=generic_weights)
        return bot, hook, {"identity_mixtures": identity_mixtures}

    if row_name == "profiled-ORACLE":
        conditioned_weights, _ = load_profiler(CONDITIONED_PATH)
        zeros = np.zeros(PARAM_DIM, dtype=np.float64)  # placeholder width
        lik = ProfilerLikelihood(conditioned_weights,
                                 seat_params={s: zeros for s in range(4)})
        identity_params = {p: param_vector(trio_ids[p]) for p in range(3)}

        def hook(seat_to_pos):
            lik.seat_params = {s: identity_params[p] for s, p in seat_to_pos.items()}

        pf = profiler_posterior_factory(lik, level=Level.FULL,
                                        n_worlds=N_WORLDS,
                                        max_draws=CHOICE_MAX_DRAWS,
                                        keep_worlds=True)
        bot = ProfiledSearchPlayer(Level.FULL, N_OUTER, N_INNER, rng,
                                   sampler_respects_voids=True,
                                   posterior_factory=pf,
                                   playout_weights=generic_weights)
        return bot, hook, {}

    raise ValueError(row_name)


def _counters(bot):
    if bot is None:
        return dict(fallbacks=0, failed_samples=0, inner_fallbacks=None,
                   inner_failed_samples=None, posterior=None)
    return dict(
        fallbacks=bot.fallbacks, failed_samples=bot.failed_samples,
        inner_fallbacks=bot.inner_fallbacks,
        inner_failed_samples=bot.inner_failed_samples,
        posterior=[bot.posterior_collapses, bot.posterior_decisions,
                  bot.posterior_worlds])


# --------------------------------------------------------------- one block
def run_block(row_name, block_idx, trio_ids, deal_seeds):
    """One MATCH: one fixed trio, `len(deal_seeds)` deals x 4 rotations."""
    is_mirror = row_name == "personality-mirror"
    bot, hook, extra = (None, None, {}) if is_mirror else \
        build_bot(row_name, trio_ids, block_rng(row_name, block_idx))

    per_deal = np.zeros(len(deal_seeds))
    q_sum = np.zeros(4)
    q_cnt = np.zeros(4)
    for i, seed in enumerate(deal_seeds):
        total = 0.0
        for rotation in range(4):
            state = deal(np.random.default_rng(seed))
            if is_mirror:
                players = [personality_player(
                    trio_ids[0],
                    opp_seat_rng(MIRROR_OPP_CONFIG_ID, seed, rotation, s))
                    for s in range(4)]
                tracked = rotation
            else:
                other_seats = sorted(s for s in range(4) if s != rotation)
                seat_to_pos = {s: p for p, s in enumerate(other_seats)}
                hook(seat_to_pos)
                players = [None, None, None, None]
                players[rotation] = bot
                for s, p in seat_to_pos.items():
                    players[s] = personality_player(
                        trio_ids[p],
                        opp_seat_rng(OPP_CONFIG_ID, seed, rotation, s))
                tracked = rotation
            final = play_game(state, players)
            assert sum(final.scores) == 26, "engine invariant broken"
            total += final.scores[tracked]
            if bot is not None and hasattr(bot, "observe_hand"):
                bot.observe_hand(list(final.history))
        per_deal[i] = total / 4.0
        q = quartile(i)
        q_sum[q] += per_deal[i]
        q_cnt[q] += 1

    diag = None
    if row_name in ("profiled-RI", "profiled-RIA"):
        diag = q_sum / np.maximum(q_cnt, 1)
    sample_weights = None
    if row_name == "profiled-RIA":
        sample_weights = extra["identity_mixtures"][0].weights.copy()
    return per_deal, _counters(bot), diag, sample_weights


# ------------------------------------------------------------------ probe
def timing_probe(n_blocks=2, n_deals_per_block=3):
    trios = block_trios(n_blocks)
    s_per_game = {}
    for name in ROW_NAMES:
        t0 = time.time()
        n_games = 0
        for b in range(n_blocks):
            seeds = [DEAL_SEED_BASE + b * BLOCK_SIZE + d
                    for d in range(n_deals_per_block)]
            run_block(name, b, trios[b], seeds)
            n_games += 4 * len(seeds)
        elapsed = time.time() - t0
        spg = elapsed / n_games
        s_per_game[name] = spg
        print(f"[probe] {name}: {spg:.3f}s/game ({elapsed:.1f}s / {n_games} "
              f"games)", flush=True)
    return s_per_game


def print_budget_projection(s_per_game, n_deals, workers):
    games = 4 * n_deals
    total = sum(spg * games for spg in s_per_game.values())
    projected_wall = total / workers
    print("=" * 72, flush=True)
    print(f"[BUDGET] projected total worker-seconds at {n_deals} deals = "
          f"{total:.0f}s ({total/3600:.2f}h serial)", flush=True)
    print(f"[BUDGET] DIVIDED BY {workers} workers -> wall clock = "
          f"{projected_wall:.0f}s ({projected_wall/3600:.2f}h) vs "
          f"{BUDGET_SECONDS/3600:.1f}h cap", flush=True)
    print("=" * 72, flush=True)
    return total, projected_wall


# ------------------------------------------------------------------ smoke
def run_smoke():
    trios = block_trios(1)
    seeds = [DEAL_SEED_BASE + d for d in range(3)]
    print(f"[smoke] block trio: {trios[0]}", flush=True)
    with open(PARTIAL, "w") as f:
        f.write(f"# smoke run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    for name in ROW_NAMES:
        t0 = time.time()
        per_deal, counters, diag, sample_weights = run_block(
            name, 0, trios[0], seeds)
        elapsed = time.time() - t0
        mean, lo, hi = bootstrap_ci(per_deal)
        print(f"[smoke] {name}: per_deal={per_deal} mean={mean:.3f} "
              f"CI=({lo:.3f},{hi:.3f}) in {elapsed:.1f}s "
              f"counters={counters}", flush=True)
        if diag is not None:
            print(f"[smoke] {name} quartile means: {diag}", flush=True)
        if sample_weights is not None:
            print(f"[smoke] {name} identity-0 mixture final weights "
                  f"(top-5): {sorted(sample_weights, reverse=True)[:5]}",
                  flush=True)
        with open(PARTIAL, "a") as f:
            f.write(name + " " + " ".join(f"{v:.6f}" for v in per_deal) + "\n")
    print("[smoke] done", flush=True)


# ------------------------------------------------------------- full driver
def _chunks(n_blocks):
    return list(range(n_blocks))


def worker(row_name, block_idx, trio_ids, deal_seeds):
    per_deal, counters, diag, sample_weights = run_block(
        row_name, block_idx, trio_ids, deal_seeds)
    return row_name, block_idx, per_deal, counters, diag, sample_weights


def total_rss_gb():
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


def _append_partial(name, block_idx, values):
    os.makedirs(RESULTS, exist_ok=True)
    with open(PARTIAL, "a") as f:
        f.write(f"{name}@{block_idx} " +
               " ".join(f"{v:.6f}" for v in values) + "\n")
        f.flush()


def run_all(n_deals, workers, trios):
    import concurrent.futures as cf
    n_blocks = n_deals // BLOCK_SIZE
    results = {name: np.zeros(n_deals) for name in ROW_NAMES}
    counters = {name: [] for name in ROW_NAMES}
    diags = {name: np.zeros((n_blocks, 4)) for name in ("profiled-RI", "profiled-RIA")}
    sample_weights_last = {}
    jobs = []
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for name in ROW_NAMES:
            for b in range(n_blocks):
                seeds = [DEAL_SEED_BASE + b * BLOCK_SIZE + d
                        for d in range(BLOCK_SIZE)]
                jobs.append(pool.submit(worker, name, b, trios[b], seeds))
        done, t0 = 0, time.time()
        for fut in cf.as_completed(jobs):
            name, b, per_deal, cnt, diag, sw = fut.result()
            results[name][b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE] = per_deal
            counters[name].append(cnt)
            if diag is not None:
                diags[name][b] = diag
            if sw is not None:
                sample_weights_last[name] = sw
            done += 1
            mem = total_rss_gb()
            print(f"[{done}/{len(jobs)}] {name} block {b} done | "
                  f"mem={mem:.1f}GB | {time.time() - t0:.0f}s elapsed",
                  flush=True)
            _append_partial(name, b, per_deal)
            if mem > MEM_LIMIT_GB:
                print(f"ABORT: memory {mem:.1f}GB exceeds {MEM_LIMIT_GB}GB",
                      flush=True)
                for j in jobs:
                    j.cancel()
                raise MemoryError("memory limit exceeded")
    return results, counters, diags, sample_weights_last


def sum_counters(clist):
    fb = sum(c["fallbacks"] for c in clist)
    fs = sum(c["failed_samples"] for c in clist)
    ifb = sum(c["inner_fallbacks"] or 0 for c in clist)
    ifs = sum(c["inner_failed_samples"] or 0 for c in clist)
    pc = [0, 0, 0]
    for c in clist:
        if c["posterior"]:
            for k in range(3):
                pc[k] += c["posterior"][k]
    return fb, fs, ifb, ifs, pc


def paired_bootstrap_ci(diffs, n_boot=10_000, rng=None):
    return bootstrap_ci(diffs, n_boot=n_boot, rng=rng)


# ------------------------------------------------------------------ output
def print_criteria(rows, diffs_vs_soft):
    print("\n" + "=" * 72, flush=True)
    print("PRE-REGISTERED CRITERIA (PHASE5_PLAN Task 6):", flush=True)
    ri_diff = diffs_vs_soft.get("profiled-RI")
    ria_diff = diffs_vs_soft.get("profiled-RIA")
    primary = False
    for label, d in (("profiled-RI", ri_diff), ("profiled-RIA", ria_diff)):
        if d is not None and d[2] < 0.0:
            primary = True
            print(f"  PRIMARY: {label} beats honest-CHOICE-soft, paired "
                  f"diff={d[0]:.3f} CI=({d[1]:.3f},{d[2]:.3f}) excludes 0 "
                  "-> PASS", flush=True)
    if not primary:
        print("  PRIMARY: neither profiled-RI nor profiled-RIA beats "
              "honest-CHOICE-soft with a paired CI excluding 0 -> FAIL",
              flush=True)

    r_mean = rows["profiled-R"]["mean"]
    ri_mean = rows["profiled-RI"]["mean"]
    ria_mean = rows["profiled-RIA"]["mean"]
    monotone = r_mean > ri_mean > ria_mean
    print(f"  SECONDARY: monotone R({r_mean:.3f}) -> RI({ri_mean:.3f}) -> "
          f"RIA({ria_mean:.3f}) -> {'PASS' if monotone else 'FAIL'} "
          "(individual steps' significance reported separately)", flush=True)

    profiled_means = {n: rows[n]["mean"] for n in
                      ("profiled-R", "profiled-RI", "profiled-RIA", "profiled-ORACLE")}
    best_name = min(profiled_means, key=profiled_means.get)
    headline = profiled_means[best_name] < BREAK_EVEN and primary
    print(f"  HEADLINE: best profiled row = {best_name} "
          f"({profiled_means[best_name]:.3f}) < {BREAK_EVEN} AND primary met "
          f"-> {'PASS' if headline else 'FAIL'}", flush=True)

    oracle_diff = diffs_vs_soft.get("profiled-ORACLE")
    gate_a = not primary
    gate_b = (oracle_diff is not None and abs(oracle_diff[0]) < GATE_ORACLE_THRESHOLD)
    print("  TWO-SIDED RL GATE:", flush=True)
    print(f"    (a) primary fails -> RL-because-we-fell-short: "
          f"{'TRIGGERED' if gate_a else 'not triggered'}", flush=True)
    if oracle_diff is not None:
        print(f"    (b) profiled-ORACLE paired edge over CHOICE-soft = "
              f"{oracle_diff[0]:.3f} pts/hand "
              f"({'<' if gate_b else '>='} {GATE_ORACLE_THRESHOLD} magnitude) "
              f"-> RL-because-reading-is-tapped-out: "
              f"{'TRIGGERED' if gate_b else 'not triggered'}", flush=True)
    verdict = "Phase 6 = Organ 3 (RL)" if (gate_a or gate_b) else \
        "Phase 6 = next inference rung / deployment (both sides cleared)"
    print(f"  GATE VERDICT: {verdict}", flush=True)
    print("=" * 72, flush=True)


def write_outputs(rows, n_deals, trios, s_per_game, elapsed, workers,
                  diags, sample_weights_last, note=None,
                  out_name="ablation5.txt", png_name="ablation5.png"):
    os.makedirs(RESULTS, exist_ok=True)
    by_name = rows
    baseline = by_name["honest-CHOICE-soft"]

    lines = [
        "open-hearts Phase 5 Task 6: THE HEADLINE ABLATION -- profiler "
        "reading/playouts/adaptation vs held-out personality trios.",
        "Lower points per hand is better; 6.5 is the symmetric break-even.",
        "",
        f"deals: {n_deals} (seeds {DEAL_SEED_BASE}.."
        f"{DEAL_SEED_BASE + n_deals - 1}), 20 match-blocks of 25 deals each "
        "vs one fixed held-out trio; identical deals/blocks across every "
        "row. CIs are 10,000 bootstrap resamples over DEALS.",
        f"GENERIC profiler: {GENERIC_PATH}",
        f"CONDITIONED profiler: {CONDITIONED_PATH}",
        f"n_outer={N_OUTER} n_inner={N_INNER} n_worlds={N_WORLDS} "
        f"max_draws={CHOICE_MAX_DRAWS}",
        f"opponent config_id (rows 1-7, shared): {OPP_CONFIG_ID}; "
        f"mirror opponent config_id: {MIRROR_OPP_CONFIG_ID}",
        "",
        "trio table (block index: held-out personality ids):",
    ]
    for b, trio in enumerate(trios):
        lines.append(f"  block {b:2d}: {trio}")
    lines += [
        "",
        "NOTE on profiled-ORACLE: kernel_profiled's playout kernel takes one "
        "GENERIC 6-tuple weight set with no per-seat parameter block, so "
        "conditioned-net-in-kernel playouts are not supported. This row is "
        "the plan's stated fallback: oracle READING (CONDITIONED net + true "
        "per-block trio params) + GENERIC playouts.",
        "",
        f"timing probe s/game per row: " +
        ", ".join(f"{n}={s:.3f}" for n, s in s_per_game.items()),
        f"wall time: {elapsed:.1f}s with {workers} worker(s)",
    ]
    if note:
        lines += ["", f"NOTE: {note}"]

    lines += [
        "",
        f"{'config':<24}{'mean':>8}{'lo95':>9}{'hi95':>9}{'fallbk':>8}"
        f"{'failsmp':>9}{'ifallbk':>9}{'ifailsmp':>9}{'pcollapse':>11}",
    ]
    for name in ROW_NAMES:
        r = rows[name]
        ifb = "-" if r["inner_fallbacks"] is None else str(r["inner_fallbacks"])
        ifs = "-" if r["inner_failed_samples"] is None else str(r["inner_failed_samples"])
        pc = "-" if r["posterior"] is None else str(r["posterior"][0])
        lines.append(f"{name:<24}{r['mean']:>8.3f}{r['lo']:>9.3f}"
                     f"{r['hi']:>9.3f}{r['fallbacks']:>8}"
                     f"{r['failed_samples']:>9}{ifb:>9}{ifs:>9}{pc:>11}")

    lines += ["", "paired per-deal diffs vs honest-CHOICE-soft (row - "
              "baseline; negative = row beats baseline), 95% paired "
              "bootstrap CI, rows 4-7:",
              f"{'config':<24}{'mean_diff':>11}{'lo95':>9}{'hi95':>9}"]
    diffs_vs_soft = {}
    for name in ("profiled-R", "profiled-RI", "profiled-RIA", "profiled-ORACLE"):
        d = rows[name]["per_deal"] - baseline["per_deal"]
        dmean, dlo, dhi = paired_bootstrap_ci(d)
        diffs_vs_soft[name] = (dmean, dlo, dhi)
        lines.append(f"{name:<24}{dmean:>11.3f}{dlo:>9.3f}{dhi:>9.3f}")

    lines += ["", "RIA/RI quartile diagnostic (mean pts/hand by "
              "position-in-block quartile 1-6/7-12/13-18/19-25, averaged "
              "over blocks):"]
    for name in ("profiled-RI", "profiled-RIA"):
        qm = diags[name].mean(axis=0)
        lines.append(f"  {name:<20}" + " ".join(f"q{i+1}={qm[i]:.3f}" for i in range(4)))

    if "profiled-RIA" in sample_weights_last:
        lines.append("")
        lines.append("RIA sanity: a block's final identity-0 mixture weights "
                     f"(top-5): {sorted(sample_weights_last['profiled-RIA'], reverse=True)[:5]}")

    with open(os.path.join(RESULTS, out_name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print_criteria(rows, diffs_vs_soft)

    names = ROW_NAMES
    means = [rows[n]["mean"] for n in names]
    err_lo = [rows[n]["mean"] - rows[n]["lo"] for n in names]
    err_hi = [rows[n]["hi"] - rows[n]["mean"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    ax.bar(names, means, yerr=[err_lo, err_hi], capsize=5, color="#4C72B0")
    ax.axhline(BREAK_EVEN, color="grey", linestyle="--", label="6.5 break-even")
    ax.set_ylabel("mean points per hand (lower is better)")
    ax.set_title(f"Phase 5 headline ablation, {n_deals} deals x 4 rotations")
    ax.legend()
    ax.tick_params(axis="x", rotation=25)
    ax2 = axes[1]
    for name, style in (("profiled-RI", "-o"), ("profiled-RIA", "-s")):
        qm = diags[name].mean(axis=0)
        ax2.plot([1, 2, 3, 4], qm, style, label=name)
    ax2.set_xlabel("quartile within block")
    ax2.set_ylabel("mean pts/hand")
    ax2.set_title("adaptation curve")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, png_name), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--deals", type=int, default=N_DEALS_TARGET)
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    if args.smoke:
        run_smoke()
        return

    if args.probe:
        s_per_game = timing_probe()
        print_budget_projection(s_per_game, args.deals, args.workers)
        print("[probe] done (no full run performed)", flush=True)
        return

    s_per_game = timing_probe()
    n_deals = args.deals
    note = None
    total, projected_wall = print_budget_projection(s_per_game, n_deals, args.workers)
    if projected_wall > BUDGET_SECONDS:
        n_deals = (n_deals // 2 // BLOCK_SIZE) * BLOCK_SIZE
        total, projected_wall = print_budget_projection(s_per_game, n_deals, args.workers)
        note = (f"projected over budget at {args.deals} deals; shrunk to "
                f"{n_deals} deals per the compute budget rule.")
        if projected_wall > BUDGET_SECONDS:
            print("\nSTOP: even the shrunk deal count is over budget. "
                  "Consult the owner. Nothing has been run.", flush=True)
            return

    trios = block_trios(n_deals // BLOCK_SIZE)
    with open(PARTIAL, "w") as f:
        f.write(f"# run start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t0 = time.time()
    results, counters, diags, sample_weights_last = run_all(n_deals, args.workers, trios)
    elapsed = time.time() - t0

    rows = {}
    for name in ROW_NAMES:
        per_deal = results[name]
        mean, lo, hi = bootstrap_ci(per_deal)
        fb, fs, ifb, ifs, pc = sum_counters(counters[name])
        inner = name not in ("personality-mirror",)
        rows[name] = {"per_deal": per_deal, "mean": mean, "lo": lo, "hi": hi,
                     "fallbacks": fb, "failed_samples": fs,
                     "inner_fallbacks": ifb if inner else None,
                     "inner_failed_samples": ifs if inner else None,
                     "posterior": pc if inner else None}
    write_outputs(rows, n_deals, trios, s_per_game, elapsed, args.workers,
                 diags, sample_weights_last, note=note)
    print(f"[done] wall time {elapsed/60:.1f} min with {args.workers} workers",
          flush=True)


if __name__ == "__main__":
    main()
