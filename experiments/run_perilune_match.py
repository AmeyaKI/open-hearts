"""B6 Tier 1: Perilune's RAW network seated in OUR harness (no-pass / no-moon Hearts).

WHAT THIS MEASURES, PLAINLY.  Perilune (JAkoliver/hearts_simulator, MIT) is an independent
neural Hearts project.  Its released networks play here through `experiments/perilune/adapter.py`
-- one forward pass and a legality-masked argmax per card, exactly their own per-deal raw
instrument -- against our shipped bot (honest-FULL 50x20, fused) and against our beginner
heuristic, on identical deals, every seat rotated.  Points per hand, lower is better.

ROWS (each is "one SUBJECT seat against three identical FIELD seats"; the subject rotates
through all four seats of every deal; value = the subject's points, averaged over rotations):

    ours-vs-3perilune        the headline direction (our bot is the minority, as C-bench / F2)
    perilune-vs-3ours        the reverse direction (their net is the minority)
    heuristic-vs-3perilune   reference rung: our beginner heuristic against their net
    perilune-vs-3heuristic   common yardstick, their side
    ours-vs-3heuristic       common yardstick, our side (same deals, same session)

`--net v5`   their 7.59M card-token transformer (md5 8a89da90): the default component of every
             later ensemble; sees the plain 550-float observation.
`--net v6.1` their CURRENT champion (md5 710c2102): v5 + two moon-defence specialists behind a
             router; sees the 882-float observation with a zero match context.

SEEDS.  Deals 100000+ (the evaluation range; pairs with every banked row).  Our seats draw
`bench_seat_seed(HARNESS_PERILUNE, row_id, net_id, deal_seed, rotation, seat)` -- one stable
stream PER GAME PER SEAT, so any single game replays alone.  Perilune's net is deterministic.
Header token `seeds=stable-v1`; a partial without the matching header is never resumed.

VARIANT HONESTY (registry wording): no-pass is in-distribution for their nets ("hold" is one
deal in four of their rotation); no-moon is OUR variant and is a handicap on THEIR side -- the
nets were trained where taking all 26 points pays.  Their strongest configuration is a SEARCH
player (Tier 2, not built); this measures the raw network only.

Usage (nothing here runs without an explicit flag):
    --smoke                      2 deals, 2 workers, offset seeds, partial deleted first
    --probe --workers N          cost probe at the launch worker count, offset seeds
    --run   --workers N          the registered run (resumable; rerun the same command)
    --report                     re-print the report from the partials
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
from openhearts.eval.seeds import HARNESS_PERILUNE, SEEDS_TOKEN, bench_seat_seed  # noqa: E402
from openhearts.eval.stats import bootstrap_ci  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.search.honest import HonestSearchPlayer  # noqa: E402
from perilune import adapter  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
DEAL_SEED_BASE = 100_000
PROBE_SEED_OFFSET = 900_000
BLOCK_SIZE = blockdriver.DEFAULT_BLOCK_SIZE
N_OUTER, N_INNER = 50, 20

# row -> (id, subject kind, field kind)
ROWS = {
    "ours-vs-3perilune": (1, "ours", "perilune"),
    "perilune-vs-3ours": (2, "perilune", "ours"),
    "heuristic-vs-3perilune": (3, "heuristic", "perilune"),
    "perilune-vs-3heuristic": (4, "perilune", "heuristic"),
    "ours-vs-3heuristic": (5, "ours", "heuristic"),
}
NET_ID = {"v5": 5, "v6.1": 61}


class _Timed:
    """Wraps a player; accumulates wall seconds and decisions into a shared dict."""

    def __init__(self, inner, sink, key):
        self.inner, self.sink, self.key = inner, sink, key

    def choose(self, view):
        t0 = time.perf_counter()
        card = self.inner.choose(view)
        self.sink[self.key + "_s"] += time.perf_counter() - t0
        self.sink[self.key + "_n"] += 1
        return card


def partial_path(row, net, tag="", kind=""):
    return os.path.join(RESULTS, f"perilune_{row}_{net}{kind}{tag}_partial.txt")


def report_path(net, tag=""):
    return os.path.join(RESULTS, f"perilune_match_{net}{tag}.txt")


def header_line(row, net, size=(N_OUTER, N_INNER)):
    sha = adapter.NETS[net][1]
    return (f"# {SEEDS_TOKEN} harness=perilune row={row} net={net} pin={adapter.PIN_COMMIT[:12]} "
            f"sha256={sha[:16]} ours=honest-FULL-{size[0]}x{size[1]}-fused")


def ensure_header(path, row, net, size=(N_OUTER, N_INNER)):
    """Fresh partial -> write the header. Existing partial -> it must carry the same header."""
    want = header_line(row, net, size)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            first = f.readline().rstrip("\n")
        if first != want:
            raise SystemExit(
                f"refusing to resume {path}: header mismatch\n  have: {first}\n  want: {want}")
        return
    with open(path, "a") as f:
        f.write(want + "\n")


def _make(kind, seed, sink, perilune_player, size):
    if kind == "ours":
        bot = HonestSearchPlayer(Level.FULL, size[0], size[1], np.random.default_rng(seed),
                                 fused=True)
        return _Timed(bot, sink, "ours"), bot
    if kind == "perilune":
        return _Timed(perilune_player, sink, "perilune"), None
    if kind == "heuristic":
        return HeuristicPlayer(), None
    raise ValueError(kind)


def play_deal(row, net, deal_seed, sink, perilune_player, size=(N_OUTER, N_INNER)):
    """One deal, four rotations. Returns (subject mean points, per-rotation subject points)."""
    row_id, subject_kind, field_kind = ROWS[row]
    per_rot = []
    for rotation in range(4):
        state = deal(np.random.default_rng(deal_seed))      # identical cards every rotation
        players, bots = [None] * 4, []
        for seat in range(4):
            kind = subject_kind if seat == rotation else field_kind
            seed = bench_seat_seed(HARNESS_PERILUNE, row_id, NET_ID[net], deal_seed, rotation, seat)
            players[seat], bot = _make(kind, seed, sink, perilune_player, size)
            if bot is not None:
                bots.append(bot)
        final = play_game(state, players)
        assert sum(final.scores) == 26, "engine invariant broken"
        per_rot.append(int(final.scores[rotation]))
        for b in bots:
            sink["fallbacks"] += int(b.fallbacks)
            sink["failed_samples"] += int(b.failed_samples)
            sink["inner_fallbacks"] += int(b.inner_fallbacks)
    return sum(per_rot) / 4.0, per_rot


def worker(name, block_idx, item_indices):
    cfg = json.loads(os.environ["PERILUNE_MATCH_CFG"])
    net, seed_base, size = cfg["net"], cfg["seed_base"], tuple(cfg["size"])
    adapter.single_thread()
    sink = {"ours_s": 0.0, "ours_n": 0, "perilune_s": 0.0, "perilune_n": 0,
            "fallbacks": 0, "failed_samples": 0, "inner_fallbacks": 0}
    uses_perilune = "perilune" in ROWS[name][1:]
    perilune_player = adapter.PeriluneRawPlayer(net) if uses_perilune else None
    t0 = time.time()
    values, rots = [], []
    for idx in item_indices:
        v, per_rot = play_deal(name, net, seed_base + idx, sink, perilune_player, size)
        values.append(v)
        rots.append(per_rot)
    return {"block_size": BLOCK_SIZE, "values": values, "rotations": rots,
            "seconds": time.time() - t0, "timing": sink}


def run_rows(rows, net, n_deals, workers, seed_base, tag="", kind="",
             size=(N_OUTER, N_INNER)):
    """`size` is the shipped 50x20 for every registered run; only the tests pass a smaller one
    (the header records it, so a small-size partial can never be resumed as a real row)."""
    os.environ["PERILUNE_MATCH_CFG"] = json.dumps(
        {"net": net, "seed_base": seed_base, "size": list(size)})
    os.makedirs(RESULTS, exist_ok=True)
    walls = {}
    for row in rows:
        path = partial_path(row, net, tag, kind)
        ensure_header(path, row, net, size)
        print(header_line(row, net, size), flush=True)
        t0 = time.time()
        blockdriver.run_blocks(row, n_deals, BLOCK_SIZE, worker, workers, path)
        walls[row] = time.time() - t0
        print(f"[{row}] {n_deals} deals in {walls[row]:.0f} s", flush=True)
    return walls


def load_row(row, net, n_deals, tag="", kind=""):
    """(per-deal values indexed by deal, merged timing) for the banked blocks, or None."""
    banked = blockdriver.load_partial(partial_path(row, net, tag, kind), row)
    if not banked:
        return None
    values, timing = {}, {}
    for b, p in banked.items():
        bs = int(p["block_size"])
        for j, v in enumerate(p["values"]):
            if b * bs + j < n_deals:
                values[b * bs + j] = float(v)
        for k, x in p["timing"].items():
            timing[k] = timing.get(k, 0) + x
    return values, timing


def _fmt(mean, lo, hi):
    return f"{mean:.3f} ({lo:.3f}, {hi:.3f})"


def _paired(a, b):
    """Bootstrap CI of a - b on the deals both rows have."""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    d = np.array([a[i] - b[i] for i in common])
    return len(common), bootstrap_ci(d)


def report(net, n_deals, tag="", kind="", out_path=None):
    lines = [f"perilune_match REPORT net={net} {time.strftime('%Y-%m-%d %H:%M')} "
             f"pin={adapter.PIN_COMMIT[:12]} variant=no-pass/no-moon points/hand (lower is better)"]
    loaded = {}
    for row in ROWS:
        got = load_row(row, net, n_deals, tag, kind)
        if got is None:
            continue
        values, timing = got
        loaded[row] = values
        v = np.array([values[i] for i in sorted(values)])
        mean, lo, hi = bootstrap_ci(v)
        field = (26.0 - v) / 3.0
        fm, flo, fhi = bootstrap_ci(field)
        mm, mlo, mhi = bootstrap_ci(field - v)
        cost = []
        for who in ("ours", "perilune"):
            if timing.get(who + "_n"):
                cost.append(f"{who} {timing[who + '_s'] / timing[who + '_n']:.4f} s/decision")
        lines.append(f"{row:24s} n={len(v):4d} subject {_fmt(mean, lo, hi)}  field seat "
                     f"{_fmt(fm, flo, fhi)}  field-subject {mm:+.3f} ({mlo:+.3f}, {mhi:+.3f})  "
                     f"[{'; '.join(cost) or 'no timed seats'}]  fallbacks="
                     f"{timing.get('fallbacks', 0)}/{timing.get('failed_samples', 0)}/"
                     f"{timing.get('inner_fallbacks', 0)}")
    pairs = [("perilune-vs-3heuristic", "ours-vs-3heuristic",
              "common yardstick: Perilune minus ours, each alone against three heuristics "
              "(positive = ours better)"),
             ("heuristic-vs-3perilune", "ours-vs-3perilune",
              "against three Perilune seats: heuristic minus ours (positive = our search adds "
              "that much over our own heuristic on this field)")]
    for a, b, text in pairs:
        if a in loaded and b in loaded:
            n, (d, lo, hi) = _paired(loaded[a], loaded[b])
            lines.append(f"    {text}: {d:+.3f} ({lo:+.3f}, {hi:+.3f}) n={n}")
    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "a") as f:
            f.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=sorted(adapter.NETS), default="v5")
    ap.add_argument("--rows", nargs="+", default=list(ROWS), choices=list(ROWS))
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tag", default="", help="filename suffix for a separate banked series")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not adapter.available():
        raise SystemExit("Perilune checkout/weights absent: run scripts/build_perilune.sh")
    if args.smoke:
        for row in args.rows:
            p = partial_path(row, args.net, args.tag, "_smoke")
            if os.path.exists(p):
                os.remove(p)
        run_rows(args.rows, args.net, 2, 2, DEAL_SEED_BASE + PROBE_SEED_OFFSET + 1000,
                 args.tag, "_smoke")
        report(args.net, 2, args.tag, "_smoke")
        return
    if args.probe:
        assert args.workers, "probe at the LAUNCH worker count: pass --workers"
        for row in args.rows:
            p = partial_path(row, args.net, args.tag, "_probe")
            if os.path.exists(p):
                os.remove(p)
        walls = run_rows(args.rows, args.net, args.deals, args.workers,
                         DEAL_SEED_BASE + PROBE_SEED_OFFSET, args.tag, "_probe")
        report(args.net, args.deals, args.tag, "_probe")
        for row, secs in walls.items():
            print(f"  {row:24s} wall {secs:6.1f} s for {args.deals} deals -> "
                  f"1000 deals ~{secs / args.deals * 1000 / 60:.1f} min at {args.workers} workers")
        return
    if args.run:
        assert args.workers, "owner-approved worker count required: --workers"
        run_rows(args.rows, args.net, args.deals, args.workers, DEAL_SEED_BASE, args.tag)
        report(args.net, args.deals, args.tag, out_path=report_path(args.net, args.tag))
        return
    if args.report:
        report(args.net, args.deals, args.tag, out_path=report_path(args.net, args.tag))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
