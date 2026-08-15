"""Phase 5 Task 3: train the profiler and score it on HELD-OUT personalities.

TRAINING-TIME script: it imports torch (and `openhearts.opponent.model`). No
play-time module does.

What it does
------------
1. Streams the Task-2 shards (`results/population_data/pop_{split}_*.npz`),
   whose rows are DECISION EVENTS: PROFILER_FEATURES_V=1 features (the
   FEATURES_V=1 333-dim layout with the hidden-hand blocks structurally zero),
   the 52-bit legal mask, the chosen card, the 4 table ids, the acting seat,
   and the acting personality's epsilon.
2. Measures torch throughput on every available device (CPU / MPS) and trains
   on whichever is faster -- both numbers land in the report.
3. Trains TWO variants with identical rows, seeds, and schedule:
     GENERIC      333 -> 256 -> 128 -> 52   (the deployment model)
     CONDITIONED  353 -> 256 -> 128 -> 52   (+ the acting personality's
                  normalized parameter vector, `opponent/params.py`)
   Loss is cross-entropy over LEGAL moves only; early stopping on val NLL.
4. Evaluates on the plan's PRE-REGISTERED HELD-OUT-PERSONALITY protocol
   (below) and prints the gates verbatim with PASS/FAIL.
5. Writes `results/profiler_train.txt` + `.png`, checkpoints and exported
   `.npz` (both variants) to `results/profiler_train/`.

THE HELD-OUT PROTOCOL (the phase's "held-out wall", Global Constraints)
----------------------------------------------------------------------
The 50 HELD-OUT personality ids from `make_population(200, 50, 314159)`
appear in NO shard (the generator asserts it). This script does not look for
them in the training data -- it GENERATES FRESH GAMES at held-out-only
tables, in a seed range disjoint from everything else in the phase
(`HELDOUT_SEED_BASE = 950000`, default 10,000 games ~= 395k decision events),
and extracts decision events with `gen_population_data.play_and_record`
itself (`pool=HELDOUT_IDS, record_heuristic=True`) rather than a copy of it,
so the featurization, table draw, per-seat rng derivation and multi-legal row
rule are literally the same code that produced the training shards. The
mirror image of the generator's assertion is checked here: every held-out
table must contain ONLY held-out ids and zero train-pool ids.

THE FOUR SCORERS (plan Task 3), all on the same held-out rows
-------------------------------------------------------------
  1. uniform-over-legal            P(c) = 1/n
  2. heuristic-match + eps=0.1     P(c) = (1-eps)*[c == heuristic's card]
                                          + eps/n,  over ALL n legal cards
  3. GENERIC profiler
  4. CONDITIONED profiler with TRUE params (the ceiling)

Baseline 2's epsilon convention, stated because Task 1 made it matter: the
mass is spread over ALL n legal cards, so a non-match still gets eps/n and
the match gets (1-eps) + eps/n. This is what CHOICE-soft assumes today: that
the opponent IS `HeuristicPlayer` and deviates uniformly 10% of the time. It
is NOT `RandomizedHeuristic`'s convention (that anchor deviates over the n-1
OTHER cards) -- the two differ, and the over-all-n form is the consistent one
here because held-out tables are personalities only, and personalities
deviate over all n (Task 1's documented convention).

DIAGNOSTIC (not a gate): the ORACLE row. Because this script replays held-out
games with the personalities' true parameters in hand, it also records each
personality's EXACT choice density at every decision,
`eps/n + (1-eps)*softmax(scores/temperature)`. That is the information-
theoretic floor -- no model can beat it in expectation -- and it is what makes
the per-noise breakdown readable: a poor log-likelihood in the eps~0.25
bucket is irreducible entropy, not model failure. Reported as a diagnostic
row and as the "headroom to oracle" column, never as a pass/fail.

METRICS
-------
  * mean log-likelihood of the chosen card (nats/decision; higher is better)
  * top-1 accuracy (fraction of decisions where the scorer's argmax is the
    card actually played). Baseline 1 has no argmax -- every legal card ties
    -- so its top-1 is reported as the expected accuracy of a uniform random
    tie-break, mean(1/n), and labelled as such.
  * per-noise-level breakdown: rows bucketed by the ACTING personality's
    epsilon.
  * calibration: EVERY (row, legal card) PAIR bucketed by the GENERIC
    profiler's predicted probability, against the realized frequency with
    which that card was the one played. (Bucketing only the chosen card's
    predicted probability would not be a calibration curve.)
  * CONDITIONED - GENERIC gap, in nats and top-1 points: the adaptation
    headroom Task 5 is trying to capture.

Usage:
  .venv/bin/python experiments/train_profiler.py --smoke
  .venv/bin/python experiments/train_profiler.py --full
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402

from openhearts.engine import cards, features  # noqa: E402
from openhearts.opponent import params as pparams  # noqa: E402
from openhearts.opponent.npz_io import N_CARDS  # noqa: E402

import gen_population_data as G  # noqa: E402  (the generator IS the extractor)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DATA_DIR = os.path.join(RESULTS, "population_data")
OUT_DIR = os.path.join(RESULTS, "profiler_train")
TXT_PATH = os.path.join(RESULTS, "profiler_train.txt")
PNG_PATH = os.path.join(RESULTS, "profiler_train.png")

# Seed hygiene (plan Global Constraints): 700000+ is population TRAINING data
# (Task 2), 100000+ is play-strength evaluation. This held-out likelihood
# evaluation gets its own disjoint range so it can never collide with either.
HELDOUT_SEED_BASE = 950_000
HELDOUT_GAMES = 10_000
SMOKE_HELDOUT_GAMES = 200

SEED_INIT = 1234        # weight init (both variants -- same init seed, and
                        # different shapes, so CONDITIONED is not GENERIC's
                        # init plus noise; documented, not a claim of
                        # identical starting weights)
SEED_SHUFFLE = 5678     # shard order + within-shard permutation, per epoch
BATCH_SIZE = 4096
LR = 1e-3
PATIENCE = 3
VAL_MAX_ROWS = 400_000  # val is ~5% of 7.89M rows; capping keeps the val set
                        # resident in RAM as float16 (~280MB at 353 dims).
                        # Deterministic head-of-stream cap, not a sample.

BASELINE2_EPS = 0.1     # the CHOICE-soft assumption under test
EPS_BUCKETS = [("eps 0.03-0.09", 0.0, 0.09), ("eps 0.09-0.15", 0.09, 0.15),
               ("eps 0.15-0.21", 0.15, 0.21), ("eps 0.21-0.25", 0.21, 1.01)]
CALIB_EDGES = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80,
                        1.0001])
GATE_TOP1_MARGIN = 0.05  # "top-1 >= baseline-2 top-1 + 5 points"
LOG_FLOOR = 1e-12        # guards log(0) for scorers that CAN assign 0; the
                         # profiler never does (softmax over legal), and
                         # baseline 2 never does either (eps/n > 0).


# --------------------------------------------------------------- shard I/O
def shard_paths(split, data_dir=DATA_DIR):
    return sorted(os.path.join(data_dir, p) for p in os.listdir(data_dir)
                  if p.startswith(f"pop_{split}_") and p.endswith(".npz"))


def param_lookup(param_lut):
    """{pid: vec} -> (sorted ids int64[K], mat float64[K, PARAM_DIM]).

    Vectorized lookup for the streaming path: a per-row list comprehension
    over `param_lut` would rebuild a 250k-row stack once per shard per epoch
    (32 shards x 20 epochs), which is the only hot spot in this script.
    `np.searchsorted` on ~250 sorted ids replaces it. Ids are sparse draws
    from 1..10M, so a dense id-indexed table is not an option.
    """
    ids = np.array(sorted(param_lut), dtype=np.int64)
    mat = np.stack([param_lut[int(i)] for i in ids])
    return ids, mat


def params_for(acting, ids, mat):
    idx = np.searchsorted(ids, np.asarray(acting, dtype=np.int64))
    assert np.all(ids[idx] == acting), "unknown personality id in a shard"
    return mat[idx]


def load_shard(path, param_lut, conditioned):
    """-> (X float16 [N, n_in], legal int64 [N], chosen int64 [N], eps [N]).

    `param_lut` is the `(ids, mat)` pair from `param_lookup`.
    """
    d = np.load(path, allow_pickle=True)
    meta = d["meta"].item()
    assert meta["features_v"] == features.FEATURES_V
    assert meta["profiler_features_v"] == 1
    X = d["profiler_features"]                    # float16 [N, NF]
    legal = d["legal_mask"].astype(np.int64)
    chosen = d["chosen_card"].astype(np.int64)
    eps = d["epsilon"].astype(np.float64)
    if conditioned:
        pid = d["personality_ids"]                 # int32 [N, 4]
        seat = d["acting_seat"].astype(np.int64)
        acting = pid[np.arange(pid.shape[0]), seat].astype(np.int64)
        P = params_for(acting, *param_lut).astype(np.float16)
        X = np.concatenate([X, P], axis=1)
    return X, legal, chosen, eps


def load_split(paths, param_lut, conditioned, max_rows=None):
    Xs, Ls, Cs, Es, n = [], [], [], [], 0
    for p in paths:
        X, legal, chosen, eps = load_shard(p, param_lut, conditioned)
        if max_rows is not None and n + X.shape[0] > max_rows:
            k = max_rows - n
            X, legal, chosen, eps = X[:k], legal[:k], chosen[:k], eps[:k]
        Xs.append(X)
        Ls.append(legal)
        Cs.append(chosen)
        Es.append(eps)
        n += X.shape[0]
        if max_rows is not None and n >= max_rows:
            break
    return (np.concatenate(Xs), np.concatenate(Ls), np.concatenate(Cs),
            np.concatenate(Es))


# ------------------------------------------------- held-out event extraction
def _true_density(player, view, legal, chosen):
    """The personality's EXACT probability of the card it played.

    Mirrors `PersonalityPlayer.choose` arithmetic (epsilon-over-all-n mixture
    plus the tempered softmax over its own scores) without consuming any
    randomness -- `_scores` is pure. Diagnostic oracle only.
    """
    n = len(legal)
    scores = player._scores(view, legal)
    w = np.exp((scores - scores.max()) / player.params.temperature)
    w = w / w.sum()
    eps = player.params.epsilon
    i = legal.index(chosen)
    return eps / n + (1.0 - eps) * float(w[i])


def extract_heldout(n_games, seed_base, progress_every=1000, log=print):
    """Fresh held-out-personality games -> decision-event arrays.

    Uses `gen_population_data.play_and_record(pool=HELDOUT_IDS,
    record_heuristic=True)` -- the SAME extractor that built the training
    shards -- then replays each game a second time (identical seeds, so
    identical play) purely to compute the oracle density, which needs the
    seated `PersonalityPlayer` objects the generator does not return.
    """
    heldout = list(G.HELDOUT_IDS)
    train_pool = set(G.TRAIN_POOL)
    Xs, Ls, Cs, Hs, Es, Ps, Os, Ns = [], [], [], [], [], [], [], []
    t0 = time.time()
    for i in range(n_games):
        seed = seed_base + i
        rows, table = G.play_and_record(seed, pool=heldout,
                                        record_heuristic=True)
        # the held-out wall, asserted in the mirror image of the generator's
        # check: only held-out ids here, and no train-pool id at all.
        assert all(t in G.HELDOUT_SET for t in table), \
            f"seed {seed}: non-held-out id at a held-out table: {table}"
        assert not (set(table) & train_pool), \
            f"seed {seed}: train-pool id leaked into a held-out table: {table}"
        k = rows["chosen_card"].shape[0]
        if k == 0:
            continue
        Xs.append(rows["profiler_features"])
        Ls.append(rows["legal_mask"])
        Cs.append(rows["chosen_card"].astype(np.int64))
        Hs.append(rows["heuristic_card"].astype(np.int64))
        Es.append(rows["epsilon"].astype(np.float64))
        seats = rows["acting_seat"].astype(np.int64)
        Ps.append(np.array([table[s] for s in seats], dtype=np.int64))
        Os.append(_oracle_for_game(seed, heldout))
        Ns.append(np.array([bin(int(m)).count("1") for m in
                            rows["legal_mask"]], dtype=np.int64))
        if progress_every and (i + 1) % progress_every == 0:
            log(f"  held-out extraction: {i + 1}/{n_games} games "
                f"({time.time() - t0:.0f}s)")
    out = dict(
        X=np.concatenate(Xs), legal=np.concatenate(Ls),
        chosen=np.concatenate(Cs), heur=np.concatenate(Hs),
        eps=np.concatenate(Es), pid=np.concatenate(Ps),
        oracle=np.concatenate(Os), n_legal=np.concatenate(Ns))
    for k, v in out.items():
        assert v.shape[0] == out["X"].shape[0], f"{k} length mismatch"
    # the extractor's own invariants, re-checked on this fresh data.
    assert np.all((out["legal"] >> out["chosen"]) & 1), \
        "a chosen card is outside its legal mask"
    assert np.all(out["n_legal"] >= 2), "a single-legal row was emitted"
    assert np.all(out["X"][:, features.OFF_HANDS + 52:
                           features.OFF_HANDS + 208] == 0.0), \
        "hidden-hand blocks are not zero in a held-out row"
    return out, time.time() - t0


def _oracle_for_game(seed, pool):
    """Replay one game to record each acting personality's true density."""
    table = G._table_for_seed(seed, pool)
    players = [G._make_player(pid, seed, s) for s, pid in enumerate(table)]
    state = G.deal(np.random.default_rng(seed))
    out = []
    while not state.is_over():
        seat = state.to_play
        view = state.view_for(seat)
        legal = cards.cards_in(view.legal_moves)
        card = players[seat].choose(view)
        if len(legal) > 1:
            out.append(_true_density(players[seat], view, legal, card))
        state.play(card)
    return np.array(out, dtype=np.float64)


# ------------------------------------------------------------------ scorers
def score_from_probs(P, chosen):
    """(mean log-likelihood, top-1 accuracy) given full [N,52] prob rows."""
    n = P.shape[0]
    pc = P[np.arange(n), chosen]
    ll = float(np.mean(np.log(np.maximum(pc, LOG_FLOOR))))
    top1 = float(np.mean(P.argmax(axis=1) == chosen))
    return ll, top1, pc


def uniform_probs(legal, n_legal):
    P = np.zeros((legal.shape[0], N_CARDS), dtype=np.float64)
    bits = ((legal[:, None] >> np.arange(N_CARDS, dtype=np.int64)[None, :])
            & 1).astype(bool)
    P[bits] = np.repeat(1.0 / n_legal, n_legal)
    return P


def heuristic_eps_probs(legal, n_legal, heur, eps=BASELINE2_EPS):
    """(1-eps)*[c == heuristic] + eps/n, spread over ALL n legal cards."""
    P = uniform_probs(legal, n_legal) * eps
    P[np.arange(P.shape[0]), heur] += (1.0 - eps)
    return P


# --------------------------------------------------------------- breakdowns
def eps_breakdown(pc_by_name, eps, chosen, argmax_by_name):
    rows = []
    for name, lo, hi in EPS_BUCKETS:
        m = (eps > lo) & (eps <= hi)
        if not m.any():
            continue
        entry = {"bucket": name, "n": int(m.sum())}
        for k in pc_by_name:
            entry[k] = {
                "ll": float(np.mean(np.log(np.maximum(pc_by_name[k][m],
                                                      LOG_FLOOR)))),
                "top1": float(np.mean(argmax_by_name[k][m] == chosen[m]))
                if argmax_by_name[k] is not None else float("nan"),
            }
        rows.append(entry)
    return rows


def calibration(P, legal, chosen):
    """Every (row, legal card) pair: predicted prob vs realized frequency."""
    bits = ((legal[:, None] >> np.arange(N_CARDS, dtype=np.int64)[None, :])
            & 1).astype(bool)
    pred = P[bits]
    picked = np.zeros_like(bits)
    picked[np.arange(P.shape[0]), chosen] = True
    real = picked[bits].astype(np.float64)
    idx = np.clip(np.searchsorted(CALIB_EDGES, pred, side="right") - 1,
                  0, len(CALIB_EDGES) - 2)
    out = []
    for b in range(len(CALIB_EDGES) - 1):
        m = idx == b
        if not m.any():
            continue
        out.append({"lo": float(CALIB_EDGES[b]), "hi": float(CALIB_EDGES[b + 1]),
                    "n": int(m.sum()), "pred": float(pred[m].mean()),
                    "real": float(real[m].mean())})
    ece = sum(r["n"] * abs(r["pred"] - r["real"]) for r in out) / len(pred)
    return out, float(ece)


# --------------------------------------------------------------------- plot
def make_plot(path, hists, scores, calib):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for name, h in hists.items():
        ax.plot(h["epoch"], h["train_nll"], marker="o", ls="--",
                label=f"{name} train")
        ax.plot(h["epoch"], h["val_nll"], marker="s", label=f"{name} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked cross-entropy (nats/decision)")
    ax.set_title("profiler learning curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = list(scores)
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [-scores[k]["ll"] for k in names], width=0.4,
           label="NLL (nats, lower better)")
    ax.bar(xs + 0.2, [scores[k]["top1"] for k in names], width=0.4,
           label="top-1 accuracy")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_title("held-out personalities: scorers")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="perfect")
    ax.plot([r["pred"] for r in calib], [r["real"] for r in calib],
            marker="o", label="GENERIC")
    ax.set_xlabel("predicted P(card)")
    ax.set_ylabel("realized frequency chosen")
    ax.set_title("calibration, all (row, legal card) pairs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 train shards / 1 val shard / 2 epochs / "
                         f"{SMOKE_HELDOUT_GAMES} held-out games")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--hidden1", type=int, default=256)
    ap.add_argument("--hidden2", type=int, default=128)
    ap.add_argument("--device", default=None,
                    help="force a device; default = measured fastest")
    ap.add_argument("--heldout-games", type=int, default=HELDOUT_GAMES)
    ap.add_argument("--data-dir", default=DATA_DIR)
    args = ap.parse_args()
    if not (args.smoke or args.full):
        ap.error("pass --smoke or --full")

    import torch
    from openhearts.opponent import model as pmodel
    from openhearts.opponent.infer import load_profiler, profiler_probs
    from openhearts.opponent.npz_io import (forward_numpy, load_npz,
                                            masked_probs_numpy)

    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    data_dir = os.path.abspath(args.data_dir)
    tr_paths = shard_paths("train", data_dir)
    va_paths = shard_paths("val", data_dir)
    epochs = args.epochs
    n_heldout_games = args.heldout_games
    if args.smoke:
        tr_paths, va_paths = tr_paths[:2], va_paths[:1]
        epochs = 2
        n_heldout_games = SMOKE_HELDOUT_GAMES
    assert tr_paths and va_paths, f"no shards found in {data_dir}"
    val_cap = 50_000 if args.smoke else VAL_MAX_ROWS

    # param lookup built ONCE for the ~203 distinct ids (never per row).
    param_lut = param_lookup(
        pparams.param_table(list(G.TRAIN_POOL) + list(G.HELDOUT_IDS)))
    print(f"data_dir={data_dir} train shards={len(tr_paths)} "
          f"val={len(va_paths)}", flush=True)
    print(f"PARAM_DIM={pparams.PARAM_DIM} "
          f"(GENERIC n_in={features.NF}, "
          f"CONDITIONED n_in={features.NF + pparams.PARAM_DIM})", flush=True)

    best_dev, dev_results = pmodel.pick_device(batch_size=args.batch_size)
    for r in dev_results:
        print(f"device {r['device']}: {r['rows_per_s']:.0f} rows/s "
              f"({r['steps']} steps x {r['batch_size']})", flush=True)
    device = args.device or best_dev
    print(f"training on device={device}", flush=True)

    # ---- held-out evaluation set (fresh games, held-out personalities only)
    print(f"extracting held-out events: {n_heldout_games} games from seed "
          f"{HELDOUT_SEED_BASE}", flush=True)
    ho, ho_s = extract_heldout(n_heldout_games, HELDOUT_SEED_BASE,
                               progress_every=1000 if not args.smoke else 0)
    n_ho = ho["X"].shape[0]
    print(f"held-out: {n_ho} decision events from {n_heldout_games} games "
          f"in {ho_s:.1f}s ({n_ho / n_heldout_games:.2f}/game)", flush=True)
    ho_pids = sorted(set(int(p) for p in ho["pid"]))
    assert set(ho_pids) <= set(G.HELDOUT_SET), "non-held-out actor in eval set"

    # The param half goes through float16 first, exactly as the training
    # shards do (their features are stored float16 and the params are cast to
    # float16 before concatenation), so evaluation inputs are bit-for-bit the
    # same construction as training inputs rather than a slightly more precise
    # variant of them.
    ho_P = params_for(ho["pid"], *param_lut).astype(np.float16)
    ho_X = {False: ho["X"].astype(np.float32),
            True: np.concatenate(
                [ho["X"].astype(np.float32), ho_P.astype(np.float32)],
                axis=1)}

    results = {}
    hists = {}
    for conditioned in (False, True):
        name = "CONDITIONED" if conditioned else "GENERIC"
        print(f"\n=== training {name} ===", flush=True)
        t0 = time.time()
        vaX, vaL, vaC, _vaE = load_split(va_paths, param_lut, conditioned,
                                         max_rows=val_cap)
        print(f"val rows {vaX.shape} loaded in {time.time() - t0:.1f}s",
              flush=True)

        def epoch_batches(ep, conditioned=conditioned):
            rng = np.random.default_rng([SEED_SHUFFLE, ep])
            order = rng.permutation(len(tr_paths))
            for si in order:
                X, legal, chosen, _e = load_shard(tr_paths[si], param_lut,
                                                  conditioned)
                perm = rng.permutation(X.shape[0])
                Xf = X.astype(np.float32)[perm]
                Lf, Cf = legal[perm], chosen[perm]
                for i in range(0, Xf.shape[0], args.batch_size):
                    yield (np.ascontiguousarray(Xf[i:i + args.batch_size]),
                           np.ascontiguousarray(Lf[i:i + args.batch_size]),
                           np.ascontiguousarray(Cf[i:i + args.batch_size]))

        model = pmodel.make_model(SEED_INIT, conditioned, args.hidden1,
                                  args.hidden2)
        t0 = time.time()
        hist = pmodel.train(model, epoch_batches, vaX, vaL, vaC,
                            device=device, epochs=epochs, lr=args.lr,
                            patience=args.patience, seed=SEED_INIT)
        train_s = time.time() - t0
        print(f"{name} trained in {train_s:.1f}s", flush=True)

        tag = name.lower()
        ckpt = os.path.join(OUT_DIR, f"profiler_{tag}.pt")
        torch.save(model.state_dict(), ckpt)
        npz_path = os.path.join(OUT_DIR, f"profiler_{tag}.npz")
        pmodel.export_npz(model, npz_path, extra={
            "seed_init": SEED_INIT, "seed_shuffle": SEED_SHUFFLE,
            "lr": args.lr, "batch_size": args.batch_size, "device": device,
            "best_epoch": hist["best_epoch"], "smoke": bool(args.smoke),
            "train_shards": len(tr_paths), "variant": tag,
            "heldout_seed_base": HELDOUT_SEED_BASE})

        lp = pmodel.predict_log_probs(model, ho_X[conditioned], ho["legal"],
                                      device)
        P = np.exp(lp)
        ll, top1, pc = score_from_probs(P, ho["chosen"])
        results[name] = {"ll": ll, "top1": top1, "pc": pc,
                         "argmax": P.argmax(axis=1), "P": P,
                         "train_s": train_s, "npz": npz_path, "ckpt": ckpt,
                         "val_rows": int(vaX.shape[0])}
        hists[name] = hist
        del vaX, vaL, vaC

    # ---- npz round-trip + numba kernel agreement + timing (GENERIC ships)
    gen_npz = results["GENERIC"]["npz"]
    w, wmeta = load_npz(gen_npz)
    (W1, b1, W2, b2, W3, b3), _m = load_profiler(gen_npz)
    n_chk = min(10_000, n_ho)
    Xc = ho_X[False][:n_chk].astype(np.float64)
    Lc = ho["legal"][:n_chk]
    ref = masked_probs_numpy(forward_numpy(w, Xc), Lc)
    got = np.zeros((n_chk, N_CARDS), dtype=np.float64)
    from openhearts.opponent.infer import profiler_probs_batch
    profiler_probs_batch(W1, b1, W2, b2, W3, b3, Xc, Lc, got)
    rel = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-3)))
    illegal_ok = bool(np.all(got[ref == 0.0] == 0.0))
    print(f"kernel vs numpy reference on {n_chk} real rows: "
          f"max_rel={rel:.3e} illegal_exact_zero={illegal_ok}", flush=True)

    # timing: single-position calls, the shape search will actually make.
    for _ in range(50):  # warm the JIT
        profiler_probs(W1, b1, W2, b2, W3, b3, Xc[0], int(Lc[0]))
    n_t = min(5000, n_chk)
    t0 = time.perf_counter()
    for i in range(n_t):
        profiler_probs(W1, b1, W2, b2, W3, b3, Xc[i], int(Lc[i]))
    us_per_call = (time.perf_counter() - t0) / n_t * 1e6

    # ---- baselines on the same held-out rows
    Pu = uniform_probs(ho["legal"], ho["n_legal"])
    ll_u, _t1u, pc_u = score_from_probs(Pu, ho["chosen"])
    top1_u = float(np.mean(1.0 / ho["n_legal"]))   # expected, ties broken u.a.r
    Ph = heuristic_eps_probs(ho["legal"], ho["n_legal"], ho["heur"])
    ll_h, top1_h, pc_h = score_from_probs(Ph, ho["chosen"])
    ll_o = float(np.mean(np.log(np.maximum(ho["oracle"], LOG_FLOOR))))

    scorers = {
        "1 uniform-over-legal": {"ll": ll_u, "top1": top1_u},
        "2 heuristic+eps=0.1": {"ll": ll_h, "top1": top1_h},
        "3 GENERIC": {"ll": results["GENERIC"]["ll"],
                      "top1": results["GENERIC"]["top1"]},
        "4 CONDITIONED (ceiling)": {"ll": results["CONDITIONED"]["ll"],
                                    "top1": results["CONDITIONED"]["top1"]},
    }
    pc_by = {"1 uniform-over-legal": pc_u, "2 heuristic+eps=0.1": pc_h,
             "3 GENERIC": results["GENERIC"]["pc"],
             "4 CONDITIONED (ceiling)": results["CONDITIONED"]["pc"],
             "oracle (diagnostic)": ho["oracle"]}
    am_by = {"1 uniform-over-legal": None,
             "2 heuristic+eps=0.1": ho["heur"],
             "3 GENERIC": results["GENERIC"]["argmax"],
             "4 CONDITIONED (ceiling)": results["CONDITIONED"]["argmax"],
             "oracle (diagnostic)": None}
    buckets = eps_breakdown(pc_by, ho["eps"], ho["chosen"], am_by)
    calib, ece = calibration(results["GENERIC"]["P"], ho["legal"],
                             ho["chosen"])

    gap_nats = results["CONDITIONED"]["ll"] - results["GENERIC"]["ll"]
    gap_top1 = results["CONDITIONED"]["top1"] - results["GENERIC"]["top1"]

    # ---- PRE-REGISTERED GATES
    g1 = (results["GENERIC"]["ll"] > ll_u) and (results["GENERIC"]["ll"] > ll_h)
    g2 = results["GENERIC"]["top1"] >= top1_h + GATE_TOP1_MARGIN

    lines = []
    A = lines.append
    A("# Phase 5 Task 3: profiler training + pre-registered held-out gates")
    A(f"# mode={'smoke' if args.smoke else 'full'} "
      f"wall_time_s={time.time() - t_start:.1f}")
    if args.smoke:
        A("# SMOKE RUN -- 2 train shards / 2 epochs / "
          f"{n_heldout_games} held-out games. The numbers below are a "
          "plumbing check, NOT the pre-registered verdict; the gates are "
          "expected to be weak or FAIL on a deliberately undertrained net.")
    A(f"# FEATURES_V={features.FEATURES_V} PROFILER_FEATURES_V=1 "
      f"PROFILER_V={wmeta['profiler_v']} NF={features.NF} "
      f"PARAM_DIM={pparams.PARAM_DIM} arch={wmeta['arch']}")
    A(f"# seeds: init={SEED_INIT} shuffle={SEED_SHUFFLE} "
      f"heldout_seed_base={HELDOUT_SEED_BASE} "
      f"population master_seed={G.MASTER_SEED}")
    A(f"# epochs_max={epochs} lr={args.lr} batch_size={args.batch_size} "
      f"patience={args.patience} optimizer=Adam "
      f"loss=cross-entropy over LEGAL moves only")
    A(f"# train shards={len(tr_paths)} val shards={len(va_paths)} "
      f"(val capped at {val_cap} rows, deterministic head of stream)")
    A(f"# HELD-OUT WALL: {n_heldout_games} FRESH games at tables drawn from "
      f"the {len(G.HELDOUT_IDS)} held-out personality ids only "
      f"(seeds {HELDOUT_SEED_BASE}..{HELDOUT_SEED_BASE + n_heldout_games - 1},"
      f" disjoint from the 700000+ training range and the 100000+ "
      f"play-strength range). {n_ho} decision events, "
      f"{len(ho_pids)} distinct held-out actors. Every table asserted "
      f"held-out-only AND train-pool-free.")
    A("#   The held-out set influences NEITHER training NOR model selection; "
      "it is extracted before training and scored exactly once, after both "
      "variants are fit. Early stopping is on TRAIN-personality val NLL, "
      "deliberately -- selecting on held-out loss would breach the wall.")
    A("#   Both variants consume IDENTICAL rows in IDENTICAL order (same "
      f"shards, same shuffle seed {SEED_SHUFFLE}, same permutation draw "
      "sequence, anchors included via pseudo-param vectors rather than "
      "dropped). That is what makes CONDITIONED - GENERIC a clean "
      "difference rather than a training-set change.")
    A("")
    A("## device measurement (training steps/s, fwd+bwd+Adam, masked-CE loss)")
    for r in dev_results:
        A(f"  {r['device']:4s}  {r['rows_per_s']:12.0f} rows/s   "
          f"({r['steps']} steps x batch {r['batch_size']} in "
          f"{r['seconds']:.3f}s)")
    A(f"  chosen device: {device}"
      f"{' (forced via --device)' if args.device else ' (fastest measured)'}")
    for name in ("GENERIC", "CONDITIONED"):
        A(f"  {name} training wall time: {results[name]['train_s']:.1f}s")
    A("  Determinism: weight init and batch order are seeded. Residual "
      "library nondeterminism is NOT ruled out -- float reductions on MPS "
      "and multithreaded CPU BLAS are not guaranteed bit-reproducible, so "
      "byte-identical re-runs are the goal, statistically-identical the "
      "documented fallback (carried from Phase 4).")
    A("")
    A("## learning curves (masked cross-entropy, nats/decision)")
    for name in ("GENERIC", "CONDITIONED"):
        h = hists[name]
        A(f"  {name}")
        A("    epoch    train_nll      val_nll     val_top1      rows")
        for i, ep in enumerate(h["epoch"]):
            A(f"    {ep:5d} {h['train_nll'][i]:12.4f} "
              f"{h['val_nll'][i]:12.4f} {h['val_top1'][i]:12.4f} "
              f"{h['n_rows'][i]:9d}")
        A(f"    best epoch={h['best_epoch']} best_val_nll="
          f"{h['best_val_nll']:.4f} (weights restored to best epoch)")
    A("")
    A("## HELD-OUT PERSONALITIES: mean log-likelihood of the chosen card "
      "(nats; higher is better) and top-1 accuracy")
    A(f"  {'scorer':<34} {'mean LL':>10} {'NLL':>10} {'top-1':>9}")
    for k in scorers:
        A(f"  {k:<34} {scorers[k]['ll']:10.4f} {-scorers[k]['ll']:10.4f} "
          f"{scorers[k]['top1']:9.4f}")
    A(f"  {'oracle: true personality density':<34} {ll_o:10.4f} "
      f"{-ll_o:10.4f} {'n/a':>9}   [DIAGNOSTIC, NOT A GATE]")
    A("  Baseline 1's top-1 is the EXPECTED accuracy of a uniform random "
      "tie-break, mean(1/n) -- a uniform scorer has no argmax.")
    A("  Baseline 2 assumes the opponent IS HeuristicPlayer and deviates "
      f"uniformly over ALL n legal cards with probability {BASELINE2_EPS}: "
      f"P(c) = (1-eps)*[c==heuristic] + eps/n. That is the CHOICE-soft "
      "assumption under test. NOTE the convention: RandomizedHeuristic (a "
      "TRAIN-only anchor) instead deviates over the n-1 OTHER cards, so the "
      "two epsilon conventions differ; held-out tables are personalities "
      "only, and Task 1 fixed personalities' epsilon as over-all-n, so the "
      "form used here is the consistent one.")
    A("  The ORACLE row is each personality's exact mixture density "
      "eps/n + (1-eps)*softmax(scores/temperature) -- the information floor. "
      "No scorer can beat it in expectation; the distance to it is what is "
      "actually learnable.")
    A(f"  headroom GENERIC -> oracle: {ll_o - results['GENERIC']['ll']:+.4f} "
      f"nats")
    A("")
    A("## ADAPTATION HEADROOM (CONDITIONED - GENERIC) -- the number Task 5 "
      "is chasing")
    A(f"  log-likelihood: {gap_nats:+.4f} nats/decision")
    A(f"  top-1:          {gap_top1 * 100:+.2f} points")
    A("  This is what perfect knowledge of an opponent's parameters buys "
      "over a single generic model. Task 5's mixture adaptation can capture "
      "at most this much (it estimates the params rather than being told).")
    A("")
    A("## PER-NOISE-LEVEL BREAKDOWN (rows bucketed by the ACTING "
      "personality's epsilon)")
    hdr = f"  {'bucket':<16}{'n':>9}"
    keys = list(pc_by)
    for k in keys:
        hdr += f"{k[:16]:>18}"
    A(hdr + "   (mean LL)")
    for b in buckets:
        line = f"  {b['bucket']:<16}{b['n']:>9}"
        for k in keys:
            line += f"{b[k]['ll']:18.4f}"
        A(line)
    A("  Higher epsilon = more of the decision is a uniform coin flip the "
      "model cannot predict, so ALL scorers (the oracle included) fall in "
      "the high-eps buckets. Compare each row against the ORACLE column, "
      "not against the low-eps buckets.")
    A("  top-1 in the same buckets:")
    for b in buckets:
        line = f"  {b['bucket']:<16}{b['n']:>9}"
        for k in keys:
            v = b[k]["top1"]
            line += ("{:>18}".format("n/a") if v != v else f"{v:18.4f}")
        A(line)
    A("")
    A("## CALIBRATION -- every (row, legal card) PAIR, bucketed by the "
      "GENERIC profiler's predicted probability")
    A(f"  {'bucket':<16}{'n pairs':>12}{'mean pred':>12}"
      f"{'realized':>12}{'gap':>10}")
    for r in calib:
        A(f"  [{r['lo']:.2f},{r['hi']:.2f})".ljust(18)
          + f"{r['n']:>10}{r['pred']:12.4f}{r['real']:12.4f}"
          f"{r['real'] - r['pred']:+10.4f}")
    A(f"  expected calibration error (pair-weighted): {ece:.4f}")
    A("")
    A("## INFERENCE (torch-free numba kernel, `opponent/infer.py`)")
    A(f"  kernel vs numpy reference on {n_chk} REAL held-out rows: "
      f"max relative diff = {rel:.3e} (gate <= 1e-6, denominator floored at "
      f"1e-3 since probabilities can be legitimately tiny)")
    A(f"  illegal cards exactly 0.0: {illegal_ok}")
    A(f"  single-position timing: {us_per_call:.2f} us/call over {n_t} real "
      f"positions (plan budget <= 20 us/call) -> "
      f"{'WITHIN' if us_per_call <= 20.0 else 'OVER'} budget")
    A("")
    A("#" * 74)
    A("## PRE-REGISTERED GATES (PHASE5_PLAN.md Task 3, stated verbatim)")
    A("##  'GENERIC must beat baselines 1 AND 2 on held-out personalities'")
    A("##  (log-likelihood strictly better, top-1 >= baseline 2 + 5 points)")
    A(f"##  GATE 1  log-likelihood strictly better than baselines 1 AND 2")
    A(f"##          GENERIC={results['GENERIC']['ll']:.4f}  "
      f"baseline1={ll_u:.4f}  baseline2={ll_h:.4f}   ->   "
      f"{'PASS' if g1 else 'FAIL'}")
    A(f"##  GATE 2  top-1 >= baseline-2 top-1 + 5 points")
    A(f"##          GENERIC={results['GENERIC']['top1'] * 100:.2f}%  "
      f"baseline2={top1_h * 100:.2f}%  "
      f"margin={(results['GENERIC']['top1'] - top1_h) * 100:+.2f} points "
      f"(need >= +5.00)   ->   {'PASS' if g2 else 'FAIL'}")
    A("##  If GENERIC cannot beat the heuristic-match baseline on "
      "personalities it")
    A("##  never saw, the plan says STOP, diagnose, owner review -- the "
      "population may")
    A("##  be too heuristic-like or the features too weak.")
    A("#" * 74)
    A("")
    A("## HOW TO READ THE GATES (framing, decided before the run, not after)")
    A(f"  * GATE 1's BINDING comparison is baseline 1 (uniform, {ll_u:.4f}), "
      f"not baseline 2 ({ll_h:.4f}). Baseline 2 loses by a huge margin "
      "because it is OVERCONFIDENT -- it stakes 0.9 on the heuristic's card "
      f"and that card is actually played {top1_h * 100:.1f}% of the time -- "
      "which is Phase 4's finding restated, not evidence that the profiler "
      "is strong. Read the margin over UNIFORM: "
      f"{results['GENERIC']['ll'] - ll_u:+.4f} nats, against a total "
      f"learnable headroom of {ll_o - ll_u:+.4f} nats from uniform to the "
      f"oracle -- i.e. this run captured "
      f"{100 * (results['GENERIC']['ll'] - ll_u) / max(ll_o - ll_u, 1e-9):.0f}"
      "% of what is available.")
    A(f"  * GATE 2 diagnostic if it FAILS: CONDITIONED reaches "
      f"{results['CONDITIONED']['top1'] * 100:.2f}% top-1, "
      f"{(results['CONDITIONED']['top1'] - top1_h) * 100:+.2f} points over "
      "baseline 2. So the +5-point margin is NOT structurally unreachable on "
      "this population -- the open question is how much of it a single "
      "generic model recovers without knowing which personality it faces, "
      "which is exactly the CONDITIONED - GENERIC gap above and exactly what "
      "Task 5's adaptation exists to close. Nothing here was tuned to chase "
      "the gate (the plan pre-registers 'not tuned until it passes').")
    A("  * The timing above is WRAPPER-INCLUSIVE (the Python dispatch in "
      "`profiler_probs`). Task 4 calls the compiled kernel from inside a "
      "numba kernel, where that overhead disappears.")

    txt = "\n".join(lines) + "\n"
    with open(TXT_PATH, "w") as f:
        f.write(txt)
    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump({"hists": hists,
                   "scorers": {k: {kk: v[kk] for kk in ("ll", "top1")}
                               for k, v in scorers.items()},
                   "oracle_ll": ll_o, "gap_nats": gap_nats,
                   "gap_top1": gap_top1, "eps_buckets": buckets,
                   "calibration": calib, "ece": ece,
                   "us_per_call": us_per_call, "kernel_max_rel": rel,
                   "devices": dev_results, "device": device,
                   "n_heldout_rows": int(n_ho),
                   "n_heldout_games": n_heldout_games,
                   "gate1_ll": bool(g1), "gate2_top1": bool(g2)}, f, indent=2)
    make_plot(PNG_PATH, hists, scorers, calib)
    print(txt)
    print(f"wrote {TXT_PATH}, {PNG_PATH}, and checkpoints/npz in {OUT_DIR}")


if __name__ == "__main__":
    main()
