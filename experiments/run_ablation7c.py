"""Phase 7C: THE SMARTER IMAGINATION -- internal fields (experts, personalities).

WHAT THIS MEASURES, PLAINLY.  Today honest search rehearses imagined futures
in which every opponent is the beginner heuristic.  7C makes the rehearsal
opponents TRAIN-side experts (search/expert_rollout.py) and asks whether the
champion takes fewer points -- against held-out experts (the ISMCTS-hard gym
7B measured), against the old Phase-5 personalities (continuity), and, via
run_cbench.py / run_xinxin_match.py `--bot expert-rollout`, against ISMCTS and
xinxin (the transfer panel).  Pre-registration: PHASE7_PLAN.md Task 7C.

ROWS (this script; deals 100000+ so every field pairs with a BANKED
incumbent row on identical cards):
    expert-rollout        ExpertRolloutSearchPlayer 50x20, pool = all 200
                          train ids, unfused (Python playouts), grouping off.
    honest-FULL-eqtime    the incumbent at `--eqtime-outer` worlds x20, fused
                          (bitwise-gated), chosen by the probe to match the
                          expert-rollout row's wall-clock.  The equal-TIME
                          control (Phase 4 precedent: the fair fight).
    (equal-WORLD control = the banked incumbent 50x20 rows: 7B honest-FULL
     for the expert field, ablation5 honest-FULL for the personality field.)
FIELDS: --field experts (held-out expert trios, 7B's blocks) | personalities
(Phase-5 held-out personality trios, ablation5's blocks).  Mirror alarms are
inherited from 7B/ablation5 and not rerun; the 7C mirror rule (print the
exact-6.5 fraction + a permutation p) applies to any NEW mirror.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from openhearts.belief.table import Level  # noqa: E402
from openhearts.engine.game import deal, play_game  # noqa: E402
from openhearts.eval import blockdriver  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.players.expert_population import train_ids  # noqa: E402
from openhearts.search.expert_rollout import ExpertRolloutSearchPlayer  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402
import run_ablation7 as a7  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
DEAL_SEED_BASE = 100_000
PROBE_SEED_OFFSET = 900_000
BLOCK_SIZE = blockdriver.DEFAULT_BLOCK_SIZE
N_INNER = 20
ROW_NAMES = ["expert-rollout", "honest-FULL-eqtime"]
ROW_CONFIG_ID = {"expert-rollout": 7300, "honest-FULL-eqtime": 7301}
FIELDS = ("experts", "personalities")
BUDGET_SECONDS = 2 * 3600


def partial_path(field, probe=False):
    tag = "_probe" if probe else ""
    return os.path.join(RESULTS, f"ablation7c_{field}{tag}_partial.txt")


def report_path(field):
    return os.path.join(RESULTS, f"ablation7c_{field}.txt")


# ------------------------------------------------------------------ fields
def _personality_field():
    import run_ablation5 as a5
    trios = a5.block_trios(a5.N_BLOCKS)          # Phase-5 held-out trios, 25-deal blocks
    def seat(pid, seed, rotation, s):
        return a5.personality_player(pid, a5.opp_seat_rng(a5.OPP_CONFIG_ID, seed, rotation, s))
    return trios, seat, 25


def _expert_field(n_deals):
    trios = a7.block_trios((n_deals + a7.TRIO_BLOCK - 1) // a7.TRIO_BLOCK)
    return trios, a7.expert_seat, a7.TRIO_BLOCK


def build_bot(row, rng, eqtime_outer):
    if row == "expert-rollout":
        return ExpertRolloutSearchPlayer(Level.FULL, 50, N_INNER, rng,
                                         rollout_ids=train_ids())
    if row == "honest-FULL-eqtime":
        assert eqtime_outer, "equal-time row needs --eqtime-outer from the probe"
        return HonestSearchPlayer(Level.FULL, int(eqtime_outer), N_INNER, rng,
                                  sampler_respects_voids=True,
                                  posterior_factory=None, fused=True)
    raise ValueError(row)


def worker(name, block_idx, item_indices):
    cfg = json.loads(os.environ["ABL7C_CFG"])
    seed_base, field, eq = cfg["seed_base"], cfg["field"], cfg["eqtime_outer"]
    trios, seat_fn, tb = (_personality_field() if field == "personalities"
                          else _expert_field(cfg["n_deals"]))
    t0 = time.time()
    bot = build_bot(name, np.random.default_rng([ROW_CONFIG_ID[name], int(block_idx)]), eq)
    values, spg = [], []
    for idx in item_indices:
        seed = seed_base + idx
        trio = trios[idx // tb]
        total = 0.0
        for rotation in range(4):
            state = deal(np.random.default_rng(seed))
            others = sorted(s for s in range(4) if s != rotation)
            players = [None] * 4
            players[rotation] = bot
            for p, s in enumerate(others):
                players[s] = seat_fn(trio[p], seed, rotation, s)
            t1 = time.time()
            final = play_game(state, players)
            spg.append(time.time() - t1)
            assert sum(final.scores) == 26
            total += final.scores[rotation]
        values.append(total / 4.0)
    return {"block_size": BLOCK_SIZE, "values": values, "seconds": time.time() - t0,
            "s_per_game": float(np.mean(spg)),
            "counters": {"fallbacks": int(bot.fallbacks), "failed_samples": int(bot.failed_samples),
                         "inner_fallbacks": int(bot.inner_fallbacks),
                         "rollout_playouts": int(getattr(bot, "rollout_playouts", 0))}}


def header(field, rows, n_deals, workers, seed_base, eq, probe):
    lines = [f"# ablation7c {'PROBE ' if probe else ''}field={field} start {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"# rows={rows} n_deals={n_deals} workers={workers} seed_base={seed_base} eqtime_outer={eq}",
             "# expert-rollout: ExpertRolloutSearchPlayer Level.FULL 50x20, pool = train ids 1-200, "
             "unfused Python playouts, grouping OFF, one identity per opponent per outer world (CRN across candidates)",
             "# honest-FULL-eqtime: HonestSearchPlayer Level.FULL <eqtime_outer>x20 fused (bitwise-gated)",
             "# variant no-pass/no-moon; paired on DEALS only with the banked incumbent rows"]
    text = "\n".join(lines)
    a7.assert_no_record_leak(text)
    return text


def run_rows(field, rows, n_deals, workers, partial, seed_base, eq, probe=False):
    os.environ["ABL7C_CFG"] = json.dumps({"seed_base": seed_base, "field": field,
                                          "eqtime_outer": eq, "n_deals": n_deals})
    hdr = header(field, rows, n_deals, workers, seed_base, eq, probe)
    print(hdr)
    with open(partial, "a") as f:
        f.write(hdr + "\n")
    out = {}
    for name in rows:
        t0 = time.time()
        blockdriver.run_blocks(name, n_deals, BLOCK_SIZE, worker, workers, partial)
        out[name] = time.time() - t0
        print(f"[{name}] {n_deals} deals in {out[name]:.0f} s")
    return out


def banked_incumbent(field):
    if field == "experts":
        rows = a7.load_rows(a7.PARTIAL)
        idx, v = a7.per_deal_vector(rows["honest-FULL"], 10_000)
        return dict(zip(idx, v))
    return a7.banked_ablation5_full()


def report(field, n_deals, partial=None, out_path=None, pair_banked=True):
    partial = partial or partial_path(field)
    lines = [f"ablation7c REPORT field={field} {time.strftime('%Y-%m-%d %H:%M')} partial={os.path.basename(partial)}"]
    inc = banked_incumbent(field) if pair_banked else {}
    for name in ROW_NAMES:
        banked = blockdriver.load_partial(partial, name)
        if not banked:
            continue
        idx, v = a7.per_deal_vector(banked, n_deals)
        mean, lo, hi = bootstrap_ci(v)
        spg = np.mean([p["s_per_game"] for p in banked.values()])
        cnt = {}
        for p in banked.values():
            for k, c in p["counters"].items():
                cnt[k] = cnt.get(k, 0) + c
        lines.append(f"{name:20s} n={len(v):4d} mean={mean:.3f} CI=({lo:.3f}, {hi:.3f}) s/game={spg:.2f} counters={cnt}")
        r = a7.paired(idx, v, inc) if inc else None
        if r:
            n, ma, mb, (d, dlo, dhi) = r
            lines.append(f"    vs banked incumbent 50x20 (equal worlds, same deals): row {ma:.3f} vs incumbent {mb:.3f}; "
                         f"row - incumbent {d:+.3f} CI=({dlo:+.3f}, {dhi:+.3f}) n={n}  (negative = row better)")
    text = "\n".join(lines)
    a7.assert_no_record_leak(text)
    print(text)
    if out_path:
        with open(out_path, "a") as f:
            f.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", choices=FIELDS, required=True)
    ap.add_argument("--rows", nargs="+", default=ROW_NAMES, choices=ROW_NAMES)
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--eqtime-outer", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        p = partial_path(args.field + "_smoke")
        if os.path.exists(p):
            os.remove(p)
        run_rows(args.field, ROW_NAMES, 2, 2, p, DEAL_SEED_BASE + PROBE_SEED_OFFSET + 1000,
                 args.eqtime_outer or 100, probe=True)
        report(args.field, 2, partial=p, pair_banked=False)
        return
    if args.probe:
        assert args.workers, "probe at the LAUNCH worker count: pass --workers"
        p = partial_path(args.field, probe=True)
        out = run_rows(args.field, args.rows, args.deals, args.workers, p,
                       DEAL_SEED_BASE + PROBE_SEED_OFFSET, args.eqtime_outer, probe=True)
        for name, secs in out.items():
            print(f"  {name:20s} wall {secs:7.1f} s ({secs / (4 * args.deals):.2f} s/game contended) "
                  f"-> 500 deals ~{secs / args.deals * 500 / 60:.0f} min, 250 ~{secs / args.deals * 250 / 60:.0f} min")
        return
    if args.run:
        assert args.workers, "owner-approved worker count required: --workers"
        run_rows(args.field, args.rows, args.deals, args.workers, partial_path(args.field),
                 DEAL_SEED_BASE, args.eqtime_outer)
        report(args.field, args.deals, out_path=report_path(args.field))
        return
    if args.report:
        report(args.field, args.deals, out_path=report_path(args.field))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
