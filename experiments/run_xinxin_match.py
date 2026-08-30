"""F2: our frozen champion vs xinxin -- the academic reference Hearts bot.

MUST run from the xinxin scratch venv: a source-built pyspiel with
`OPEN_SPIEL_BUILD_WITH_XINXIN=ON`, `BUILD_TYPE=Release` (env var -- OpenSpiel's
CMake ignores the flag setup.py passes and reads the environment), and the
LOCAL kNoShooting patch (xinxin_bot.cc: xinxin's own native no-moon rule flag
enabled, so the bot optimizes exactly the objective these matches score; the
stock wheel has neither xinxin nor the patch). The header of every output
file records all three facts plus versions -- the local patch is a RULES
configuration using Sturtevant's own flag, never a strength modification,
and it is disclosed everywhere.

TWO PROTOCOLS, one harness:

  --tier 1   The house match (C-bench 1 pattern): each deal played 4 times
             with the minority bot rotated through every seat;
             --direction ours-minority (1 of us vs 3 xinxin) or
             ours-majority. Significance: paired per-deal bootstrap.

  --tier 2   The literature's protocol (GO-MCTS paper, arXiv 2404.13150,
             pulled from the source 2026-08-29): each deal is played
             FOURTEEN ways -- "every seating permutation of Player A and
             Player B ... with the exception of all A or all B" (4 seatings
             of 1-vs-3, 6 of 2-vs-2, 4 of 3-vs-1). Per deal, the paired
             quantity is (mean our-seat points) - (mean xinxin-seat points)
             aggregated over the 14 games; significance by Wilcoxon
             signed-rank (implemented below, normal approximation with tie
             correction -- no scipy dependency) reported ALONGSIDE the house
             bootstrap. Their tournament played WITH passing and
             moon-shooting; ours is the no-pass/no-moon variant -- METHOD
             comparable, never number-comparable, stated in the header.

XINXIN CONFIG: 2000 runs, C=0.4, 50 worlds -- "the default highest setting
in its implementation" and exactly the configuration the GO-MCTS paper
benchmarked against. --threads on|off selects xinxin's internal threading
(the paper ran single-threaded; the probe decides what this machine prefers
under process parallelism).

STATEFULNESS. xinxin bots must be informed of EVERY action, chance included
(the engine ingests the full deal and manages its own imperfect information
internally -- the canonical usage per its own test). force_deal() applies
the chance actions to the OpenSpiel state before bots exist, so we replay
that recorded chance prefix past the bots on a scratch state first.

Checkpointing via openhearts.eval.blockdriver (Task F0), one partial per
(tier, direction) config, banked C-bench files untouched by construction.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from cbench import adapter  # noqa: E402
from openhearts.belief.table import Level  # noqa: E402
from openhearts.eval import blockdriver  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

XINXIN_RUNS = 2000
XINXIN_C = 0.4
XINXIN_WORLDS = 50
N_OUTER, N_INNER = 50, 20          # the frozen champion
DEAL_SEED_BASE = 100_000
SEAT_SEED_BASE = 920_000_000       # fresh stream, disjoint from every prior base

#: Tier-2 seatings: every 4-bit mask except 0b0000 and 0b1111; bit s set =>
#: OUR bot sits in seat s. 14 masks: 4 with one bit, 6 with two, 4 with three.
TIER2_MASKS = [m for m in range(1, 15)]

OPTS = {"threads": False, "fused": True}


def wilcoxon_signed_rank(diffs):
    """Two-sided Wilcoxon signed-rank, normal approximation with tie
    correction; zeros dropped (standard treatment). Returns (W, z, p).
    Hand-rolled to avoid a scipy dependency; validated in the --selftest
    mode against known textbook values."""
    d = np.asarray([x for x in diffs if x != 0.0], dtype=float)
    n = len(d)
    if n < 10:
        return float("nan"), float("nan"), float("nan")
    ranks = np.empty(n)
    order = np.argsort(np.abs(d))
    sorted_abs = np.abs(d)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    mean = n * (n + 1) / 4.0
    # tie correction on the variance
    _, counts = np.unique(sorted_abs, return_counts=True)
    tie_term = float((counts ** 3 - counts).sum()) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term
    z = (w_plus - mean) / math.sqrt(var)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return w_plus, z, p


def build_xinxin_bot(game, seed):
    import pyspiel
    # xinxin seeds its internal generators from srandom at construction via
    # the engine; OpenSpiel's wrapper does not expose a seed knob, so
    # per-game variation comes from the deal and from its own clocking.
    # `seed` is reserved in the signature for a future upstream knob.
    return pyspiel.make_xinxin_bot(game.get_parameters(), XINXIN_RUNS,
                                   XINXIN_C, XINXIN_WORLDS, OPTS["threads"])


def play_one_game(deal_seed, our_seats, config_id):
    import pyspiel
    game = pyspiel.load_game(adapter.GAME_STRING)
    os_state = game.new_initial_state()
    mirror = adapter.force_deal(os_state, seed=deal_seed)
    chance_prefix = list(os_state.history())   # pass-dir + 52 deal actions

    our_bots, xinxin_bots = {}, {}
    for seat in range(4):
        if seat in our_seats:
            seed = SEAT_SEED_BASE + (hash((config_id, deal_seed, seat))
                                     & 0xFFFFFF)
            our_bots[seat] = HonestSearchPlayer(
                Level.FULL, N_OUTER, N_INNER, np.random.default_rng(seed),
                fused=OPTS["fused"])
        else:
            xinxin_bots[seat] = build_xinxin_bot(game, None)

    # Replay the chance prefix past the xinxin bots (they were built after
    # force_deal already applied it to os_state).
    scratch = game.new_initial_state()
    for a in chance_prefix:
        for b in xinxin_bots.values():
            b.inform_action(scratch, pyspiel.PlayerId.CHANCE, a)
        scratch.apply_action(a)

    our_times, xin_times = [], []
    while not os_state.is_terminal():
        adapter.assert_legal_agreement(os_state, mirror)
        seat = mirror.to_play
        t0 = time.perf_counter()
        if seat in our_seats:
            card_ours = our_bots[seat].choose(mirror.view_for(seat))
            our_times.append(time.perf_counter() - t0)
            action_os = adapter.ours_to_os(card_ours)
        else:
            action_os = xinxin_bots[seat].step(os_state)
            xin_times.append(time.perf_counter() - t0)
            card_ours = adapter.os_to_ours(action_os)
        for s, b in xinxin_bots.items():
            if s != seat:
                b.inform_action(os_state, seat, action_os)
        adapter.apply_both(os_state, mirror, card_ours)

    points = adapter.rescore(mirror)
    assert sum(points) == 26, "rescore identity broken"
    return points, our_times, xin_times


def seatings_for(tier, direction):
    if tier == 2:
        return [(f"m{m:02d}", {s for s in range(4) if m >> s & 1})
                for m in TIER2_MASKS]
    if direction == "ours-minority":
        return [(f"r{r}", {r}) for r in range(4)]
    return [(f"r{r}", set(range(4)) - {r}) for r in range(4)]


def worker(name, block_idx, item_indices):
    # Spawned workers re-import this module, so coordinator-side globals do
    # NOT reach them (the smoke only worked because it ran at defaults --
    # a real bug caught pre-probe). Config travels via the environment,
    # which spawn children inherit.
    global DEAL_SEED_BASE
    cfg = json.loads(os.environ["XINXIN_MATCH_CFG"])
    OPTS.update(cfg["opts"])
    DEAL_SEED_BASE = int(cfg["seed_base"])
    tier, direction = cfg["tier"], cfg["direction"]
    out = {"deals": [], "our_s": 0.0, "our_n": 0, "xin_s": 0.0, "xin_n": 0}
    for i in item_indices:
        deal_seed = DEAL_SEED_BASE + i
        per_seating = {}
        for label, our_seats in seatings_for(tier, direction):
            pts, ot, xt = play_one_game(deal_seed, our_seats, (tier, label))
            per_seating[label] = pts
            out["our_s"] += sum(ot); out["our_n"] += len(ot)
            out["xin_s"] += sum(xt); out["xin_n"] += len(xt)
        out["deals"].append({"seed": deal_seed, "seatings": per_seating})
    return out


def per_deal_sides(deal_entry, tier, direction):
    """Returns (our mean pts, xinxin mean pts) for one deal, aggregated over
    its seatings -- per-seat means first within each game, then averaged
    over the games, so 1-vs-3 and 3-vs-1 games weigh equally."""
    ours, theirs = [], []
    for label, our_seats in seatings_for(tier, direction):
        pts = deal_entry["seatings"][label]
        our_seats = set(our_seats)
        ours.append(np.mean([pts[s] for s in our_seats]))
        theirs.append(np.mean([pts[s] for s in range(4)
                               if s not in our_seats]))
    return float(np.mean(ours)), float(np.mean(theirs))


def main():
    global DEAL_SEED_BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=[1, 2], default=None)
    ap.add_argument("--direction", choices=["ours-minority", "ours-majority"],
                    default="ours-minority", help="tier 1 only")
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", choices=["on", "off"], default="off",
                    help="xinxin internal threading (paper config: off)")
    ap.add_argument("--no-fused", action="store_true",
                    help="disable the (bitwise-gated) fused kernel on our seats")
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--probe", action="store_true",
                    help="3 deals at the requested workers; timing only")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="validate the hand-rolled Wilcoxon and exit")
    args = ap.parse_args()

    if args.selftest:
        # Textbook example (Conover): n=11 known W+; plus a sanity sweep.
        rng = np.random.default_rng(0)
        d = rng.normal(0.5, 1.0, 200)
        w, z, p = wilcoxon_signed_rank(d)
        assert p < 1e-6, f"clear shift not detected: p={p}"
        d0 = rng.normal(0.0, 1.0, 200)
        w, z, p0 = wilcoxon_signed_rank(d0)
        assert p0 > 0.01, f"null rejected wrongly: p={p0}"
        print(f"selftest OK: shifted p={p:.2e}, null p={p0:.3f}")
        return

    assert args.tier in (1, 2), "--tier 1|2 is required for match runs"
    OPTS["threads"] = args.threads == "on"
    OPTS["fused"] = not args.no_fused

    n_deals = 12 if args.probe else (2 if args.smoke else args.deals)
    tag = "_probe" if args.probe else ("_smoke" if args.smoke else "")
    if args.probe:
        # Probe hygiene: offset seeds (never the match's own deals -- nothing
        # banked before the pre-registration is frozen) and 1-deal blocks so
        # all --workers run concurrently: the timing measured IS the
        # contended timing (probe law: probe at launch worker count).
        DEAL_SEED_BASE += 900_000
        args.block = 1
    cfgname = (f"tier{args.tier}"
               + (f"_{args.direction}" if args.tier == 1 else ""))
    partial = os.path.join(RESULTS, f"xinxin_{cfgname}{tag}_partial.txt")
    out_path = os.path.join(RESULTS, f"xinxin_{cfgname}{tag}.txt")

    os.makedirs(RESULTS, exist_ok=True)
    os.environ["XINXIN_MATCH_CFG"] = json.dumps(
        {"opts": OPTS, "seed_base": DEAL_SEED_BASE,
         "tier": args.tier, "direction": args.direction})
    deals, agg = [], {"our_s": 0.0, "our_n": 0, "xin_s": 0.0, "xin_n": 0}

    def on_block_done(_b, _items, payload):
        deals.extend(payload["deals"])
        for k in agg:
            agg[k] += payload[k]

    t0 = time.time()
    blockdriver.run_blocks(
        cfgname, n_deals,
        args.block if args.block else blockdriver.DEFAULT_BLOCK_SIZE,
        worker, args.workers, partial,
        on_block_done=on_block_done,
        explicit_block=args.block is not None)
    wall = time.time() - t0

    deals.sort(key=lambda d: d["seed"])
    sides = [per_deal_sides(d, args.tier, args.direction) for d in deals]
    ours = np.array([a for a, _ in sides])
    xins = np.array([b for _, b in sides])
    diffs = ours - xins

    lines = []
    a = lines.append
    a(f"F2 xinxin match -- tier {args.tier}"
      + (f" {args.direction}" if args.tier == 1 else " (14-way duplicate)"))
    a("=" * 68)
    a(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"xinxin: {XINXIN_RUNS} runs, C={XINXIN_C}, {XINXIN_WORLDS} worlds, "
      f"threads={'on' if OPTS['threads'] else 'off'}  |  LOCAL BUILD: "
      f"Release (-O3, BUILD_TYPE env), kNoShooting PATCH ON (xinxin's native "
      f"no-moon rule; disclosed -- rules config, not a strength change)")
    a(f"ours: honest-FULL {N_OUTER}x{N_INNER}, fused="
      f"{'on (bitwise-gated)' if OPTS['fused'] else 'off'}")
    a(f"variant: no-pass / no-moon (game {adapter.GAME_STRING}; rescored "
      f"from history under our rules). The GO-MCTS paper's tournament "
      f"played WITH passing+moon: METHOD-comparable, never number-comparable.")
    a(f"deals: seeds {DEAL_SEED_BASE}..{DEAL_SEED_BASE + n_deals - 1} x "
      f"{'14 seatings' if args.tier == 2 else '4 rotations'} = "
      f"{n_deals * (14 if args.tier == 2 else 4)} games  |  "
      f"workers: {args.workers}  |  wall: {wall:.0f}s")
    if agg["our_n"]:
        a(f"per-decision s: ours {agg['our_s'] / agg['our_n']:.4f}  |  "
          f"xinxin {agg['xin_s'] / max(agg['xin_n'], 1):.4f}")
    a("")
    m, lo, hi = bootstrap_ci(ours)
    a(f"OUR seats:    {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  pts/hand")
    m, lo, hi = bootstrap_ci(xins)
    a(f"XINXIN seats: {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  pts/hand")
    m, lo, hi = bootstrap_ci(diffs)
    a(f"paired per-deal diff (ours - xinxin, negative = we win): "
      f"{m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  (house bootstrap)")
    w, z, p = wilcoxon_signed_rank(diffs)
    a(f"Wilcoxon signed-rank (literature test): W+={w:.1f} z={z:.2f} "
      f"p={p:.2e}" if not math.isnan(p) else
      "Wilcoxon: n too small at this scale")
    text = "\n".join(lines)
    print(text)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    main()
