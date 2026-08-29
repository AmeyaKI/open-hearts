"""Task A3 (PHASE7_PLAN.md): is 50x20 still the right operating point?

WHY, PLAINLY. The champion's search size (50 outer worlds x 20 inner
re-determinization samples) was chosen when a decision cost ~208 ms. The
Phase-2.8 fused kernel made decisions 5.3x cheaper -- and nobody ever
re-asked what bigger search buys at the new prices. The C0 probe says the
answer might not be "nothing": at 50 worlds the bot disagrees with itself
39% of the time on the same decision, at 200 worlds only 29% -- the argmin
is still substantially sampling noise at the shipping size. This script
prices the operating points on the HEURISTIC field (continuity with
Phase 1's sweep); the ISMCTS field runs through run_cbench.py (which
already takes --n-outer/--n-inner) in the pyspiel scratch venv.

WHAT IT DOES. For each config in the grid, one honest-FULL seat (fused
kernel ON -- bitwise-gated, so semantics identical, only wall clock moves)
plays 250 deals x 4 rotations against three heuristics, paired on the same
deal seeds across configs. Reports per-config mean + 95% CI, PAIRED
per-deal diff vs the 50x20 incumbent, and measured s/decision -- the
points-per-second curve the adoption decision (7D, license + fresh R0 +
exploitability) will consume. A3 itself adopts nothing.

SANITY BRACKET (not a bitwise gate): the 50x20 row must land within its own
CI of the published honest-FULL 3.253 (Phase 2, same deals). Seat rngs here
are a fresh derivation, so equality is statistical, not bitwise -- stated
here so nobody mistakes a 3.2-vs-3.3 wobble for a regression.

Checkpointing via openhearts.eval.blockdriver (Task F0), one partial file
per config, none of them touching any banked reference file.

Usage:
    .venv/bin/python experiments/run_world_sweep.py --smoke
    .venv/bin/python experiments/run_world_sweep.py --deals 250 --workers 12
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openhearts.belief.table import Level  # noqa: E402
from openhearts.engine.game import deal  # noqa: E402
from openhearts.eval import blockdriver  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RESULTS, "world_sweep.txt")

LEVEL = Level.FULL
CONFIGS = [(50, 20), (100, 20), (200, 20), (100, 50)]   # (n_outer, n_inner)
INCUMBENT = (50, 20)
DEAL_SEED_BASE = 100_000          # the house comparison range
SEAT_SEED_BASE = 910_000_000      # fresh stream, disjoint from every prior base
PUBLISHED_INCUMBENT = 3.253       # Phase-2 honest-FULL vs heuristics (sanity bracket)


def seat_seed(deal_seed, rotation):
    return SEAT_SEED_BASE + deal_seed * 4 + rotation


def cfg_name(cfg):
    return f"{cfg[0]}x{cfg[1]}"


def play_deal(cfg, deal_seed):
    """One deal, 4 rotations; returns (mean tracked pts, decision seconds,
    decision count). The tracked seat is the only search seat."""
    n_outer, n_inner = cfg
    total = 0.0
    dec_s = 0.0
    dec_n = 0
    for rotation in range(4):
        state = deal(np.random.default_rng(deal_seed))
        players = [HeuristicPlayer() for _ in range(4)]
        bot = HonestSearchPlayer(
            LEVEL, n_outer, n_inner,
            np.random.default_rng(seat_seed(deal_seed, rotation)),
            fused=True)
        players[rotation] = bot
        while not state.is_over():
            seat = state.to_play
            view = state.view_for(seat)
            if seat == rotation:
                t0 = time.perf_counter()
                card = bot.choose(view)
                dec_s += time.perf_counter() - t0
                dec_n += 1
            else:
                card = players[seat].choose(view)
            state.play(card)
        assert sum(state.scores) == 26, "engine invariant broken"
        total += state.scores[rotation]
    return total / 4.0, dec_s, dec_n


def worker(name, block_idx, item_indices):
    cfg = next(c for c in CONFIGS if cfg_name(c) == name)
    values, dec_s, dec_n = [], 0.0, 0
    for i in item_indices:
        v, s, n = play_deal(cfg, DEAL_SEED_BASE + i)
        values.append(v)
        dec_s += s
        dec_n += n
    return {"values": values, "dec_s": dec_s, "dec_n": dec_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=250)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_deals = 4 if args.smoke else args.deals
    tag = "_smoke" if args.smoke else ""

    os.makedirs(RESULTS, exist_ok=True)
    res = {}
    t_start = time.time()
    for cfg in CONFIGS:
        name = cfg_name(cfg)
        vals = np.zeros(n_deals)
        agg = {"dec_s": 0.0, "dec_n": 0}

        def on_block_done(b, items, payload, vals=vals, agg=agg):
            vals[items[0]:items[0] + len(payload["values"])] = payload["values"]
            agg["dec_s"] += payload["dec_s"]
            agg["dec_n"] += payload["dec_n"]

        partial = os.path.join(RESULTS, f"world_sweep_{name}{tag}_partial.txt")
        blockdriver.run_blocks(
            name, n_deals,
            args.block if args.block else blockdriver.DEFAULT_BLOCK_SIZE,
            worker, args.workers, partial,
            on_block_done=on_block_done,
            explicit_block=args.block is not None)
        res[name] = (vals, agg["dec_s"] / max(agg["dec_n"], 1))

    lines = []
    a = lines.append
    a("Phase 7 Task A3 -- operating-point sweep at post-fusion prices")
    a("=" * 66)
    a(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"field: 1 honest-FULL seat (fused kernel ON, bitwise-gated) vs 3x "
      f"HeuristicPlayer")
    a(f"deals: seeds {DEAL_SEED_BASE}..{DEAL_SEED_BASE + n_deals - 1} x 4 "
      f"rotations, paired across configs  |  workers: {args.workers}  |  "
      f"seat seeds: f(deal,rotation), base {SEAT_SEED_BASE}")
    a(f"wall: {time.time() - t_start:.0f}s  |  JIT: "
      f"{'off' if os.environ.get('OPENHEARTS_NO_JIT') else 'on'}")
    a("")
    a(f"{'config':>8} | {'pts/hand':>8} | {'95% CI':>18} | "
      f"{'paired vs 50x20':>22} | {'s/decision':>10}")
    inc = res[cfg_name(INCUMBENT)][0]
    rng = np.random.default_rng(0)
    for cfg in CONFIGS:
        name = cfg_name(cfg)
        v, spd = res[name]
        m, lo, hi = bootstrap_ci(v, rng=rng)
        if cfg == INCUMBENT:
            paired = "(incumbent)"
        else:
            dm, dlo, dhi = bootstrap_ci(v - inc, rng=rng)
            paired = f"{dm:+.3f} [{dlo:+.3f},{dhi:+.3f}]"
        a(f"{name:>8} | {m:>8.3f} | [{lo:.3f}, {hi:.3f}]   | {paired:>22} | "
          f"{spd:>10.4f}")
    a("")
    inc_mean = float(inc.mean())
    a(f"sanity bracket: 50x20 mean {inc_mean:.3f} vs published 3.253 "
      f"(statistical, not bitwise -- fresh seat-rng derivation; see header)")
    a("adoption is NOT decided here: any changed operating point is a new bot")
    a("version -> 7D license check + fresh R0 + exploitability regression")
    a("(pre-registered: more worlds = less accidental noise = chin may rise).")
    text = "\n".join(lines)
    print(text)
    out = OUT.replace(".txt", f"{tag}.txt")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
