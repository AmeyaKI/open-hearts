"""Phase 7B: THE EXPERT EXAM (and the ORACLE / ORACLE-SIR diagnostics).

WHAT THIS MEASURES, PLAINLY.  The champion (honest-FULL, unchanged) plays
the 50 SEALED held-out experts (players/expert_population.py, ids 201-250)
on deal seeds 100000+, the deals where it already played the Phase-5
personality gym (results/ablation5_partial.txt) and the stock ISMCTS bot
(results/cbench_ours-minority_partial.txt).  "How hard is the new exam" is
therefore a PAIRED three-way comparison on identical cards, and the two
banked rows cost nothing to rerun.  Pre-registration: PHASE7_PLAN.md Task 7B
(frozen 2026-08-27 expectations + the 2026-09-05 pre-launch notes D1-D8).

ROWS (config ids stable; append only):
    honest-FULL         incumbent, Level.FULL 50x20, fused kernel ON
                        (bitwise-gated vs the unfused path), no posterior;
                        three held-out experts per 25-deal match block.
    expert-mirror       alarm: four copies of the block's first expert,
                        independent tie rngs; must bracket 6.5.
    expert-ORACLE       honest-FULL whose outer worlds come from the EXACT
                        expert-policy likelihood (true held-out records,
                        per seat), N=50 draws, survivors only, collapse ->
                        counted fallback.  Diagnostic.
    expert-ORACLE-SIR   same likelihood, pool grown to the ESS target or the
                        cap, resampled to N equal weights (A1's law).
                        Diagnostic, mandatory per A1.

MATCH BLOCKS.  Deal index -> trio is a pure function (25 consecutive deals
share one trio, trio drawn from derive_seed(DOMAIN_SEAT_ROTATION, ...)), so
every row sees the same people on the same cards and the paired per-deal
bootstrap is unaffected.  Checkpoints are the F0 blockdriver's 5-deal
blocks inside those 25-deal trio blocks (results/ablation7_partial.txt,
append-only, resumable).  Bot rng per 5-deal block from
(row config id, block idx); expert tie rng per game from
derive_seed(DOMAIN_POLICY_TIES, deal_seed, rotation, seat, expert_id).

THE WALL.  Held-out records are instantiated in-process only.  This script
logs trio IDS, never a parameter field; `assert_no_record_leak` guards the
header text.

MODES
  --smoke                plumbing on own `_smoke` paths (2 deals, all rows).
  --probe --workers W    timing at the LAUNCH worker count on OFFSET seeds
                         (+900000; never banks); prints s/game, projection.
  --run --rows ... --workers W --deals N   the real thing (owner go first).
  --report               summarize banked partials vs every banked row.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from openhearts.belief.expert_oracle import (  # noqa: E402
    ExpertLikelihood, expert_oracle_factory, expert_sir_factory)
from openhearts.belief.table import Level  # noqa: E402
from openhearts.engine.game import deal, play_game  # noqa: E402
from openhearts.eval import blockdriver  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.players.expert import ExpertPlayer  # noqa: E402
from openhearts.players.expert_population import (  # noqa: E402
    DOMAIN_POLICY_TIES, DOMAIN_SEAT_ROTATION, MASTER_SEED_EXPERT_V1,
    OPTIONAL_WEIGHTS, derive_seed, heldout_ids, sample_expert)
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402
from openhearts.search.sir import SIRRecorder, merge_payloads  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
PARTIAL = os.path.join(RESULTS, "ablation7_partial.txt")
REPORT = os.path.join(RESULTS, "ablation7.txt")
BANKED_ABLATION5 = os.path.join(RESULTS, "ablation5_partial.txt")
BANKED_CBENCH = os.path.join(RESULTS, "cbench_ours-minority_partial.txt")

DEAL_SEED_BASE = 100_000
PROBE_SEED_OFFSET = 900_000
TRIO_BLOCK = 25
BLOCK_SIZE = blockdriver.DEFAULT_BLOCK_SIZE     # 5
N_OUTER, N_INNER = 50, 20
N_WORLDS = 50
ORACLE_MAX_DRAWS = 2000
SIR_M_CAP = 2000
BUDGET_SECONDS = 2 * 3600
BREAK_EVEN = 6.5

ROW_NAMES = ["honest-FULL", "expert-mirror", "expert-ORACLE", "expert-ORACLE-SIR"]
ROW_CONFIG_ID = {name: 7100 + i for i, name in enumerate(ROW_NAMES)}
EXAM_ROWS = ["honest-FULL", "expert-mirror"]
DIAG_ROWS = ["expert-ORACLE", "expert-ORACLE-SIR"]


# ------------------------------------------------------------------ trios
def block_trios(n_trio_blocks):
    held = heldout_ids()
    trios, seen = [], set()
    b = 0
    while len(trios) < n_trio_blocks:
        rng = np.random.default_rng(
            derive_seed(DOMAIN_SEAT_ROTATION, MASTER_SEED_EXPERT_V1, b))
        idx = rng.choice(len(held), size=3, replace=False)
        trio = tuple(int(held[int(i)]) for i in idx)
        b += 1
        if frozenset(trio) in seen:
            continue
        seen.add(frozenset(trio))
        trios.append(trio)
    return trios


def trio_for_deal(deal_idx, trios):
    return trios[deal_idx // TRIO_BLOCK]


def expert_seat(expert_id, deal_seed, rotation, seat):
    return ExpertPlayer(np.random.default_rng(
        derive_seed(DOMAIN_POLICY_TIES, deal_seed, rotation, seat, expert_id)),
        sample_expert(expert_id))


def block_rng(row_name, block_idx):
    return np.random.default_rng([ROW_CONFIG_ID[row_name], int(block_idx)])


# ------------------------------------------------------------------ bots
def build_bot(row_name, rng, recorder):
    """-> (bot, hook(seat_params_by_seat))."""
    if row_name == "honest-FULL":
        bot = HonestSearchPlayer(Level.FULL, N_OUTER, N_INNER, rng,
                                 sampler_respects_voids=True,
                                 posterior_factory=None, fused=True)
        return bot, (lambda sp: None)
    lik = ExpertLikelihood()
    if row_name == "expert-ORACLE":
        pf = expert_oracle_factory(lik, Level.FULL, N_WORLDS, ORACLE_MAX_DRAWS,
                                   recorder=recorder)
    elif row_name == "expert-ORACLE-SIR":
        pf = expert_sir_factory(lik, Level.FULL, N_WORLDS, m_cap=SIR_M_CAP,
                                recorder=recorder)
    else:
        raise ValueError(row_name)
    bot = HonestSearchPlayer(Level.FULL, N_OUTER, N_INNER, rng,
                             sampler_respects_voids=True,
                             posterior_factory=pf, fused=True)

    def hook(seat_params):
        lik.seat_params = dict(seat_params)
    return bot, hook


def _counters(bot):
    if bot is None:
        return {}
    return {"fallbacks": int(bot.fallbacks),
            "failed_samples": int(bot.failed_samples),
            "inner_fallbacks": int(bot.inner_fallbacks),
            "posterior_collapses": int(bot.posterior_collapses),
            "posterior_decisions": int(bot.posterior_decisions),
            "posterior_worlds": int(bot.posterior_worlds)}


# ------------------------------------------------------------------ worker
_CFG = {}


def worker(name, block_idx, item_indices):
    """One 5-deal checkpoint block of one row: `item_indices` are deal
    indices; seeds are DEAL_SEED_BASE (+ probe offset) + index."""
    cfg = json.loads(os.environ["ABL7_CFG"])
    seed_base = cfg["seed_base"]
    trios = block_trios(cfg["n_trio_blocks"])
    t0 = time.time()
    is_mirror = name == "expert-mirror"
    recorder = SIRRecorder() if name in DIAG_ROWS else None
    bot, hook = (None, None) if is_mirror else build_bot(name, block_rng(name, block_idx), recorder)
    values = []
    dec_time = []
    for idx in item_indices:
        seed = seed_base + idx
        trio = trio_for_deal(idx, trios)
        total = 0.0
        for rotation in range(4):
            state = deal(np.random.default_rng(seed))
            if is_mirror:
                players = [expert_seat(trio[0], seed, rotation, s) for s in range(4)]
            else:
                others = sorted(s for s in range(4) if s != rotation)
                seat_params = {s: sample_expert(trio[p]) for p, s in enumerate(others)}
                hook(seat_params)
                players = [None] * 4
                players[rotation] = bot
                for p, s in enumerate(others):
                    players[s] = expert_seat(trio[p], seed, rotation, s)
            t1 = time.time()
            final = play_game(state, players)
            dec_time.append(time.time() - t1)
            assert sum(final.scores) == 26, "engine invariant broken"
            total += final.scores[rotation]
        values.append(total / 4.0)
    payload = {"block_size": BLOCK_SIZE, "values": values,
               "counters": _counters(bot), "seconds": time.time() - t0,
               "s_per_game": float(np.mean(dec_time)),
               "trio_ids": [list(trio_for_deal(i, trios)) for i in item_indices[:1]],
               "sir": recorder.payload() if recorder is not None else None}
    return payload


# ------------------------------------------------------------------ driver
def header(rows, n_deals, workers, seed_base, probe):
    trios = block_trios((n_deals + TRIO_BLOCK - 1) // TRIO_BLOCK)
    lines = [
        f"# ablation7 {'PROBE ' if probe else ''}start {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# rows={rows} n_deals={n_deals} workers={workers} seed_base={seed_base} "
        f"trio_block={TRIO_BLOCK} checkpoint_block={BLOCK_SIZE}",
        f"# champion=honest-FULL Level.FULL {N_OUTER}x{N_INNER} fused=ON (bitwise-gated) "
        f"posterior=None; ORACLE N={N_WORLDS} max_draws={ORACLE_MAX_DRAWS}; "
        f"SIR N={N_WORLDS} m_cap={SIR_M_CAP} ess_target={N_WORLDS} systematic",
        f"# population=expert v1 master={MASTER_SEED_EXPERT_V1} held-out ids 201-250; "
        f"trios by block (ids only): {trios}",
        "# variant no-pass/no-moon; paired on DEALS only (search rngs not paired across rows)",
    ]
    text = "\n".join(lines)
    assert_no_record_leak(text)
    return text


def assert_no_record_leak(text):
    for w in OPTIONAL_WEIGHTS + ("tie_mode", "void_window", "exit_count"):
        assert w not in text, f"held-out wall: parameter field {w!r} in output"


def run_rows(rows, n_deals, workers, partial, seed_base, probe=False):
    os.environ["ABL7_CFG"] = json.dumps({
        "seed_base": seed_base,
        "n_trio_blocks": (n_deals + TRIO_BLOCK - 1) // TRIO_BLOCK})
    hdr = header(rows, n_deals, workers, seed_base, probe)
    print(hdr)
    with open(partial, "a") as f:
        f.write(hdr + "\n")
    out = {}
    for name in rows:
        t0 = time.time()
        res = blockdriver.run_blocks(name, n_deals, BLOCK_SIZE, worker, workers,
                                     partial)
        out[name] = (res, time.time() - t0)
        print(f"[{name}] {n_deals} deals in {time.time() - t0:.0f} s")
    return out


def load_rows(partial):
    rows = {}
    for name in ROW_NAMES:
        banked = blockdriver.load_partial(partial, name)
        if banked:
            rows[name] = banked
    return rows


def per_deal_vector(banked, n_deals):
    vals = {}
    for b, payload in banked.items():
        for j, v in enumerate(payload["values"]):
            vals[b * BLOCK_SIZE + j] = float(v)
    idx = sorted(i for i in vals if i < n_deals)
    return idx, np.array([vals[i] for i in idx])


def banked_ablation5_full():
    """Phase-5 honest-FULL per-deal values, keyed by deal index (25-blocks)."""
    out = {}
    if not os.path.exists(BANKED_ABLATION5):
        return out
    with open(BANKED_ABLATION5) as f:
        for line in f:
            if not line.startswith("honest-FULL@"):
                continue
            head, rest = line.split(" ", 1)
            b = int(head.split("@")[1])
            for j, v in enumerate(rest.split()):
                out[b * 25 + j] = float(v)
    return out


def banked_cbench_ours():
    out = {}
    if not os.path.exists(BANKED_CBENCH):
        return out
    with open(BANKED_CBENCH) as f:
        for line in f:
            if line.startswith("our@"):
                head, v = line.split()
                out[int(head[4:]) - DEAL_SEED_BASE] = float(v)
    return out


def paired(a_idx, a, b_map):
    common = [i for i in a_idx if i in b_map]
    if not common:
        return None
    da = np.array([a[a_idx.index(i)] for i in common])
    db = np.array([b_map[i] for i in common])
    return len(common), da.mean(), db.mean(), bootstrap_ci(da - db)


def report(partial, n_deals, out_path=None, smoke=False, pair_banked=True):
    rows = load_rows(partial)
    lines = [f"ablation7 REPORT {time.strftime('%Y-%m-%d %H:%M')} partial={os.path.basename(partial)}"]
    vec = {}
    for name, banked in rows.items():
        idx, v = per_deal_vector(banked, n_deals)
        vec[name] = (idx, v)
        mean, lo, hi = bootstrap_ci(v)
        secs = sum(p.get("seconds", 0.0) for p in banked.values())
        spg = np.mean([p.get("s_per_game", float("nan")) for p in banked.values()])
        cnt = {}
        for p in banked.values():
            for k, c in p.get("counters", {}).items():
                cnt[k] = cnt.get(k, 0) + c
        lines.append(f"{name:18s} n={len(v):4d} mean={mean:.3f} CI=({lo:.3f}, {hi:.3f}) "
                     f"s/game={spg:.2f} wall={secs:.0f}s counters={cnt}")
        if name == "expert-mirror":
            ok = lo <= BREAK_EVEN <= hi
            lines.append(f"    MIRROR ALARM {'OK' if ok else 'FIRED'}: 6.5 "
                         f"{'inside' if ok else 'OUTSIDE'} CI (alarm evaluated 1 time)")
        if name in DIAG_ROWS:
            payloads = [p["sir"] for p in banked.values() if p.get("sir")]
            if payloads:
                merged = merge_payloads(payloads)
                lines.append(f"    per-trick survivors/ESS: {summarize_sir(merged)}")
    if "honest-FULL" in vec:
        idx, full = vec["honest-FULL"]
        for name in DIAG_ROWS:
            if name in vec:
                r = paired(idx, full, dict(zip(*vec[name])))
                if r:
                    n, ma, mb, (d, lo, hi) = r
                    lines.append(f"reading value {name} = FULL - row: {d:+.3f} "
                                 f"CI=({lo:+.3f}, {hi:+.3f}) n={n}  (positive = row beats FULL)")
        r = paired(idx, full, banked_ablation5_full()) if pair_banked else None
        if r:
            n, ma, mb, (d, lo, hi) = r
            lines.append(f"EXAM vs Phase-5 personality gym (banked ablation5 honest-FULL, same deals): "
                         f"experts {ma:.3f} vs personalities {mb:.3f}; paired diff {d:+.3f} "
                         f"CI=({lo:+.3f}, {hi:+.3f}) n={n}")
        r = paired(idx, full, banked_cbench_ours()) if pair_banked else None
        if r:
            n, ma, mb, (d, lo, hi) = r
            lines.append(f"EXAM vs ISMCTS@1000 (banked C-bench ours-minority, same deals): "
                         f"experts {ma:.3f} vs ISMCTS {mb:.3f}; paired diff {d:+.3f} "
                         f"CI=({lo:+.3f}, {hi:+.3f}) n={n}")
    text = "\n".join(lines)
    assert_no_record_leak(text)
    print(text)
    if out_path and not smoke:
        with open(out_path, "a") as f:
            f.write(text + "\n")
    return text


def summarize_sir(merged):
    """Per-trick means from summed payloads: pool/survivors m, ESS, distinct
    worlds, cap-hit rate; plus the recorder's own firing counts."""
    parts = []
    pt = merged["per_trick"]
    for trick in sorted(pt, key=int):
        r = pt[trick]
        n = max(1.0, r["n"])
        parts.append(f"t{trick}:m={r['m']/n:.0f}/ess={r['ess_pool']/n:.0f}"
                     f"/distinct={r['distinct_resampled']/n:.0f}/cap={r['cap_hits']/n:.0%}")
    parts.append(f"| decisions={merged['n_decisions']} cap_hits={merged['cap_hits']}")
    return " ".join(parts)


def probe(rows, n_deals, workers):
    partial = os.path.join(RESULTS, "ablation7_probe_partial.txt")
    out = run_rows(rows, n_deals, workers, partial,
                   DEAL_SEED_BASE + PROBE_SEED_OFFSET, probe=True)
    print(f"\nPROBE at {workers} workers, {n_deals} deals x 4 rotations per row:")
    for name, (res, secs) in out.items():
        games = 4 * n_deals
        proj500 = secs / n_deals * 500
        proj250 = secs / n_deals * 250
        print(f"  {name:18s} wall {secs:7.1f} s ({secs / games:.2f} s/game contended) "
              f"-> 500 deals ~{proj500 / 60:.0f} min, 250 deals ~{proj250 / 60:.0f} min "
              f"{'(over 2 h budget at 500)' if proj500 > BUDGET_SECONDS else ''}")
    print("Watchdog: experiments/watchdog.py --partial results/ablation7_partial.txt "
          "--block-eta <s per 5-deal block from above> --k 3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rows", nargs="+", default=EXAM_ROWS, choices=ROW_NAMES)
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    if args.smoke:
        partial = os.path.join(RESULTS, "ablation7_smoke_partial.txt")
        for p in (partial,):
            if os.path.exists(p):
                os.remove(p)
        run_rows(ROW_NAMES, 2, 2, partial, DEAL_SEED_BASE + PROBE_SEED_OFFSET + 1000,
                 probe=True)
        report(partial, 2, smoke=True, pair_banked=False)   # offset seeds: never pair with banked rows
        return
    if args.probe:
        assert args.workers, "probe at the LAUNCH worker count: pass --workers"
        probe(args.rows, args.deals, args.workers)
        return
    if args.run:
        assert args.workers, "owner-approved worker count required: --workers"
        run_rows(args.rows, args.deals, args.workers, PARTIAL, DEAL_SEED_BASE)
        report(PARTIAL, args.deals, REPORT)
        return
    if args.report:
        report(PARTIAL, args.deals, REPORT)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
