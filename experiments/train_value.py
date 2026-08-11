"""Phase 4 Task 3: train the value net and score it against honest baselines.

TRAINING-TIME script: it imports torch (and `openhearts.value.model`). No
play-time module does.

What it does
------------
1. Streams the Task-2 shards (`results/value_data/value_{split}_*.npz`),
   reconstructing the 4-output ROTATED target vector from the shards' scalar
   per-row targets (see `rotated_targets`), and asserts column 0 is the
   evaluated seat (the shard's own `targets`) exactly.
2. Measures torch throughput on every available device (CPU / MPS) and trains
   on whichever is faster -- both numbers land in the report.
3. Trains MLP 333->128->64->4 (ReLU, MSE, Adam) with early stopping on val.
4. Scores four pre-registered baselines on the SAME held-out TEST rows,
   all as MSE on the evaluated seat's remaining points (output/column 0):
     1 constant  -- train-set mean remaining points at that PLY depth
                    (ply, not trick: the shards' finest depth key).
     2 linear    -- ridge regression on the same 333 features, closed form
                    from streamed X'X / X'y in float64, lambda=1e-3.
     3 playout   -- ONE heuristic playout (the kernel, i.e. exactly what
                    search uses today) from the reconstructed position.
     4 20x playout -- see the DEGENERACY note below.
5. Writes `results/value_train.txt` and `results/value_train.png`, plus the
   torch checkpoint and exported `.npz` under `results/value_train/`.

TWO RECORDED PLAN DEVIATIONS / CORRECTIONS
------------------------------------------
(a) BASELINE 4 IS DEGENERATE. Heuristic playouts are deterministic given a
    state, so 20 playouts from one position are 20 identical playouts.
    Baseline 4 is therefore numerically IDENTICAL to baseline 3 and is
    reported as such -- no synthetic variance is invented. What baseline 4
    was reaching for (the variance a search world-BATCH averages over) comes
    from re-sampling *different determinized worlds* from the belief table,
    which is Task 5 territory and needs a belief state this task does not
    have. Stated in the output file too.
(b) BASELINE 3 READS THE ANSWER KEY ON 60% OF TEST ROWS. The training data's
    heuristic-mix games (seed % 5 < 3) were played BY HeuristicPlayer, and
    the kernel playout is a bitwise-exact port of that same heuristic. So on
    those rows "one heuristic playout from here" reproduces the recorded
    continuation card-for-card and its MSE is exactly 0. Verified empirically
    (and asserted in this script, where it doubles as the correctness gate on
    position reconstruction). Test is 60% heuristic / 40% randomized (the
    split key %100 and the mix key %5 are not independent), so the overall
    baseline-3 number measures data-generation coupling, not estimator
    quality. The pre-registered PASS/FAIL is still reported VERBATIM on the
    overall number; the mix-split columns are reported next to it, and the
    randomized-mix column is the closest honest estimator comparison
    available here. This is an evaluation-design artifact for the lead/owner
    to weigh -- it is deliberately NOT used to redefine the gate.

Usage:
  .venv/bin/python experiments/train_value.py --smoke
  .venv/bin/python experiments/train_value.py --full
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402

from openhearts.engine import features, kernel  # noqa: E402
from openhearts.engine.game import deal  # noqa: E402

import gen_value_data as G  # noqa: E402  (player mix + split rules live there)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
DATA_DIR = os.path.join(RESULTS, "value_data")
OUT_DIR = os.path.join(RESULTS, "value_train")
TXT_PATH = os.path.join(RESULTS, "value_train.txt")
PNG_PATH = os.path.join(RESULTS, "value_train.png")

SEED_INIT = 1234        # weight init
SEED_SHUFFLE = 5678     # shard order + within-shard shuffling
RIDGE_LAMBDA = 1e-3
BATCH_SIZE = 4096
LR = 1e-3
PATIENCE = 3
BUCKETS = [("tricks 1-4", 0, 16), ("tricks 5-9", 16, 36),
           ("tricks 10-13", 36, 52)]


# ------------------------------------------------------------------- shards
def shard_paths(split, data_dir=DATA_DIR):
    return sorted(os.path.join(data_dir, p) for p in os.listdir(data_dir)
                  if p.startswith(f"value_{split}_") and p.endswith(".npz"))


def rotated_targets(targets, seat_index, game_ids, ply_index):
    """Scalar per-row targets -> [N, 4] rotated-seat targets.

    Shard rows come in ply-major/seat-minor groups of 4 (one whole game is
    appended at a time, 52*4 rows), row (i, s) carrying absolute seat s's
    remaining points with FEATURES rotated to seat s. So the rotated target
    vector of row (i, s) is T[i, (s + r) % 4] for r in 0..3, and column 0 is
    the row's own target -- asserted below, which is the task's "column 0 is
    the evaluated seat" verification against the shard schema.
    """
    n = targets.shape[0]
    assert n % 4 == 0, "shard rows are not whole (ply, 4-seat) groups"
    tg = targets.reshape(-1, 4)
    si = seat_index.reshape(-1, 4)
    assert np.all(si == np.arange(4, dtype=si.dtype)[None, :]), \
        "seat_index groups are not [0,1,2,3]"
    assert np.all(game_ids.reshape(-1, 4) ==
                  game_ids.reshape(-1, 4)[:, :1]), "game_id varies in group"
    assert np.all(ply_index.reshape(-1, 4) ==
                  ply_index.reshape(-1, 4)[:, :1]), "ply_index varies in group"
    out = np.empty((tg.shape[0], 4, 4), dtype=np.float32)
    for s in range(4):
        for r in range(4):
            out[:, s, r] = tg[:, (s + r) % 4]
    out = out.reshape(n, 4)
    assert np.array_equal(out[:, 0], targets), \
        "rotated column 0 is not the evaluated seat's target"
    return out


def load_shard(path):
    d = np.load(path, allow_pickle=True)
    meta = d["meta"].item()
    assert meta["features_v"] == features.FEATURES_V
    X = d["features"]                       # float16 [N, NF]
    y = d["targets"].astype(np.float32)     # [N]
    seat = d["seat_index"].astype(np.int64)
    ply = d["ply_index"].astype(np.int64)
    gid = d["game_ids"].astype(np.int64)
    Y = rotated_targets(y, seat, gid, ply)
    return X, Y, y, seat, ply, gid


def load_split(paths):
    Xs, Ys, ys, seats, plys, gids = [], [], [], [], [], []
    for p in paths:
        X, Y, y, seat, ply, gid = load_shard(p)
        Xs.append(X.astype(np.float32))
        Ys.append(Y)
        ys.append(y)
        seats.append(seat)
        plys.append(ply)
        gids.append(gid)
    return (np.concatenate(Xs), np.concatenate(Ys), np.concatenate(ys),
            np.concatenate(seats), np.concatenate(plys), np.concatenate(gids))


# ---------------------------------------------------------------- baselines
def fit_constant(train_paths):
    """Mean remaining points per PLY depth (0..51), fit on train."""
    s = np.zeros(52, dtype=np.float64)
    c = np.zeros(52, dtype=np.float64)
    for p in train_paths:
        d = np.load(p, allow_pickle=True)
        np.add.at(s, d["ply_index"].astype(np.int64),
                  d["targets"].astype(np.float64))
        np.add.at(c, d["ply_index"].astype(np.int64), 1.0)
    return s / np.maximum(c, 1.0)


def fit_ridge(train_paths, lam=RIDGE_LAMBDA):
    """Closed-form ridge on the 333 features + intercept, streamed float64."""
    nf = features.NF
    XtX = np.zeros((nf + 1, nf + 1), dtype=np.float64)
    Xty = np.zeros(nf + 1, dtype=np.float64)
    for p in train_paths:
        d = np.load(p, allow_pickle=True)
        X = d["features"].astype(np.float64)
        y = d["targets"].astype(np.float64)
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        XtX += X1.T @ X1
        Xty += X1.T @ y
    reg = lam * np.eye(nf + 1)
    reg[nf, nf] = 0.0  # do not penalise the intercept
    return np.linalg.solve(XtX + reg, Xty)


def predict_ridge(beta, X):
    return (X.astype(np.float64) @ beta[:-1] + beta[-1]).astype(np.float64)


def playout_predictions(game_seeds, progress=None):
    """Baseline 3: one heuristic playout from every ply of every game.

    Replays each game from its deal seed with the SAME players the generator
    used (`gen_value_data._players_for`), snapshots the kernel arrays before
    each ply, and runs `kernel.playout_to_end` on copies. One playout gives
    all four seats' points-from-here at once.

    Returns {seed: float64[52, 4]} indexed by ABSOLUTE seat, plus a dict of
    the recorded true remaining points for the same keys (used for the
    heuristic-mix exactness assertion).
    """
    preds, truths, mixes = {}, {}, {}
    out_c = np.zeros(52, dtype=np.int64)
    out_s = np.zeros(52, dtype=np.int64)
    for k, seed in enumerate(game_seeds):
        players, mix = G._players_for(seed)
        state = deal(np.random.default_rng(seed))
        snaps = []
        while not state.is_over():
            hands, scores, tc, ts, tl, pm = kernel._to_arrays(state)
            snaps.append((hands, scores, tc, ts, tl, pm, state.to_play,
                          state.hearts_broken, state.trick_number))
            seat = state.to_play
            state.play(players[seat].choose(state.view_for(seat)))
        final = np.array(state.scores, dtype=np.int64)
        assert len(snaps) == 52 and final.sum() == 26
        pred = np.zeros((52, 4), dtype=np.float64)
        true = np.zeros((52, 4), dtype=np.float64)
        for i, (hands, scores, tc, ts, tl, pm, tp, hb, tn) in enumerate(snaps):
            h, sc = hands.copy(), scores.copy()
            c, s = tc.copy(), ts.copy()
            kernel.playout_to_end(h, tp, pm, c, s, tl, hb, tn, sc,
                                  out_c, out_s)
            pred[i] = sc - scores
            true[i] = final - scores
        if mix == "heuristic":
            # Correctness gate AND the degeneracy finding: the generator's
            # HeuristicPlayer and the kernel playout are the same policy, so
            # the playout must reproduce the recorded continuation exactly.
            assert np.array_equal(pred, true), (
                f"seed {seed}: heuristic-mix playout != recorded outcome; "
                f"position reconstruction is wrong")
        preds[seed], truths[seed], mixes[seed] = pred, true, mix
        if progress and (k + 1) % progress == 0:
            print(f"  playouts: {k + 1}/{len(game_seeds)} games", flush=True)
    return preds, truths, mixes


def gather_playout(preds, gids, plys, seats):
    out = np.empty(gids.shape[0], dtype=np.float64)
    for i in range(gids.shape[0]):
        out[i] = preds[int(gids[i])][int(plys[i]), int(seats[i])]
    return out


# ------------------------------------------------------------------ metrics
def mse(pred, true):
    d = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return float(np.mean(d * d))


def bucket_mses(pred, true, plys):
    out = {}
    for name, lo, hi in BUCKETS:
        m = (plys >= lo) & (plys < hi)
        out[name] = mse(pred[m], true[m]) if m.any() else float("nan")
    return out


# --------------------------------------------------------------------- plot
def make_plot(path, hist, bucket_table, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(hist["epoch"], hist["train_mse"], marker="o", label="train MSE")
    ax.plot(hist["epoch"], hist["val_mse"], marker="s", label="val MSE")
    ax.axvline(hist["best_epoch"], color="gray", ls="--", lw=1,
               label=f"best epoch {hist['best_epoch']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE (all 4 rotated outputs)")
    ax.set_title("value net v1 learning curves")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = [b[0] for b in BUCKETS]
    w = 0.8 / len(order)
    xs = np.arange(len(names))
    for i, key in enumerate(order):
        vals = [bucket_table[key][n] for n in names]
        ax.bar(xs + i * w - 0.4 + w / 2, vals, width=w, label=key)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("test MSE (evaluated seat's remaining points)")
    ax.set_title("test MSE by trick bucket, all predictors")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 train shards / 1 val / 1 test shard, 2 epochs")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--device", default=None,
                    help="force a device; default = measured fastest")
    ap.add_argument("--hidden1", type=int, default=128)
    ap.add_argument("--hidden2", type=int, default=64)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--tag", default="v1",
                    help="output name tag: value_<tag>.{pt,npz}, "
                         "value_train_<tag>.txt/.png when != v1")
    args = ap.parse_args()
    if not (args.smoke or args.full):
        ap.error("pass --smoke or --full")

    import torch
    from openhearts.value import model as vmodel
    from openhearts.value.npz_io import forward_numpy, load_npz

    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    tr_paths = shard_paths("train")
    va_paths = shard_paths("val")
    te_paths = shard_paths("test")
    epochs = args.epochs
    if args.smoke:
        tr_paths, va_paths, te_paths = tr_paths[:2], va_paths[:1], te_paths[:1]
        epochs = 2
    assert tr_paths and va_paths and te_paths, "no shards found"

    print(f"train shards={len(tr_paths)} val={len(va_paths)} "
          f"test={len(te_paths)}", flush=True)

    # ---- device measurement (both numbers reported, faster one used)
    best_dev, dev_results = vmodel.pick_device(batch_size=args.batch_size)
    for r in dev_results:
        print(f"device {r['device']}: {r['rows_per_s']:.0f} rows/s "
              f"({r['steps']} steps x {r['batch_size']})", flush=True)
    device = args.device or best_dev
    print(f"training on device={device}", flush=True)

    # ---- val / test in memory
    t0 = time.time()
    vaX, vaY, _vay, _vas, _vap, _vag = load_split(va_paths)
    teX, teY, tey, teseat, teply, tegid = load_split(te_paths)
    print(f"loaded val {vaX.shape} test {teX.shape} in {time.time()-t0:.1f}s",
          flush=True)

    # ---- streaming epoch batches (deterministic order + shuffle)
    def epoch_batches(ep):
        rng = np.random.default_rng([SEED_SHUFFLE, ep])
        order = rng.permutation(len(tr_paths))
        for si in order:
            X, Y, _y, _s, _p, _g = load_shard(tr_paths[si])
            perm = rng.permutation(X.shape[0])
            Xf = X.astype(np.float32)[perm]
            Yf = Y[perm]
            for i in range(0, Xf.shape[0], args.batch_size):
                yield (np.ascontiguousarray(Xf[i:i + args.batch_size]),
                       np.ascontiguousarray(Yf[i:i + args.batch_size]))

    model = vmodel.make_model(SEED_INIT, args.hidden1, args.hidden2)
    t0 = time.time()
    hist = vmodel.train(model, epoch_batches, vaX, vaY, device=device,
                        epochs=epochs, lr=args.lr, patience=args.patience,
                        seed=SEED_INIT)
    train_s = time.time() - t0
    print(f"training done in {train_s:.1f}s", flush=True)

    ckpt = os.path.join(OUT_DIR, f"value_{args.tag}.pt")
    torch.save(model.state_dict(), ckpt)
    npz_path = os.path.join(OUT_DIR, f"value_{args.tag}.npz")
    vmodel.export_npz(model, npz_path, extra={
        "seed_init": SEED_INIT, "seed_shuffle": SEED_SHUFFLE, "lr": args.lr,
        "batch_size": args.batch_size, "device": device,
        "best_epoch": hist["best_epoch"], "smoke": bool(args.smoke),
        "train_shards": len(tr_paths)})

    # ---- npz round-trip check vs torch (>=1k rows)
    w, wmeta = load_npz(npz_path)
    n_chk = min(4096, teX.shape[0])
    ref = vmodel.predict(model, teX[:n_chk], device).astype(np.float64)
    got = forward_numpy(w, teX[:n_chk])
    abs_err = float(np.abs(got - ref).max())
    # The reference is torch FLOAT32 (its own representation error is ~6e-8
    # relative), and targets live on 0..26 points, so the relative metric
    # floors the denominator at 1 point: this measures agreement in points,
    # not float32 rounding on outputs that happen to sit near zero. The naive
    # |diff|/|ref| is reported too, over |ref| > 1e-3, for completeness.
    floor = 1e-3
    m = np.abs(ref) > floor
    rel_err = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)))
    rel_err_naive = float(np.max(np.abs(got[m] - ref[m]) / np.abs(ref[m])))
    print(f"npz round-trip on {n_chk} rows: max_abs={abs_err:.3e} "
          f"max_rel(points-scaled)={rel_err:.3e} "
          f"max_rel(naive)={rel_err_naive:.3e}", flush=True)

    # ---- predictions on TEST (evaluated seat = column 0 / shard `targets`)
    true = tey.astype(np.float64)
    pred_net = vmodel.predict(model, teX, device)[:, 0].astype(np.float64)
    const = fit_constant(tr_paths)
    pred_const = const[teply]
    t0 = time.time()
    beta = fit_ridge(tr_paths)
    ridge_s = time.time() - t0
    pred_lin = predict_ridge(beta, teX)

    test_seeds = sorted(set(int(g) for g in tegid))
    t0 = time.time()
    preds_po, truths_po, mixes = playout_predictions(
        test_seeds, progress=200 if len(test_seeds) > 200 else None)
    playout_s = time.time() - t0
    pred_po = gather_playout(preds_po, tegid, teply, teseat)
    # cross-check: recorded targets must equal the replay's own outcome
    true_replay = gather_playout(truths_po, tegid, teply, teseat)
    assert np.max(np.abs(true_replay - true)) < 1e-6, \
        "replayed game outcomes disagree with the shard targets"

    is_heur = np.array([mixes[int(g)] == "heuristic" for g in tegid])

    preds = {"MLP (value net)": pred_net, "linear (ridge)": pred_lin,
             "constant (per-ply mean)": pred_const,
             "baseline 3: 1 heuristic playout": pred_po,
             "baseline 4: mean of 20 playouts (== baseline 3, degenerate)":
                 pred_po}
    order = list(preds)
    overall = {k: mse(v, true) for k, v in preds.items()}
    buckets = {k: bucket_mses(v, true, teply) for k, v in preds.items()}
    heur_only = {k: mse(v[is_heur], true[is_heur]) for k, v in preds.items()}
    rand_only = {k: mse(v[~is_heur], true[~is_heur]) for k, v in preds.items()}

    train_mse_eval = mse(
        vmodel.predict(model, vaX, device)[:, 0].astype(np.float64),
        vaY[:, 0].astype(np.float64))

    # train-vs-test gap: net MSE on a train shard (col 0) vs test
    Xtr0, Ytr0, ytr0, _s0, _p0, _g0 = load_shard(tr_paths[0])
    net_train_mse = mse(vmodel.predict(model, Xtr0.astype(np.float32),
                                       device)[:, 0], ytr0.astype(np.float64))

    # ---- pre-registered criteria, verbatim
    c1 = (overall["MLP (value net)"] < overall["linear (ridge)"] <
          overall["constant (per-ply mean)"])
    c2 = overall["MLP (value net)"] < overall["baseline 3: 1 heuristic playout"]
    c2_rand = (rand_only["MLP (value net)"] <
               rand_only["baseline 3: 1 heuristic playout"])

    lines = []
    A = lines.append
    A("# Phase 4 Task 3: value net v1 training + pre-registered baselines")
    A(f"# mode={'smoke' if args.smoke else 'full'} "
      f"wall_time_s={time.time() - t_start:.1f}")
    if args.smoke:
        A("# SMOKE RUN -- 2 train shards / 2 epochs. The numbers below are a "
          "plumbing check, NOT the pre-registered verdict; the criteria "
          "section is expected to FAIL on a deliberately undertrained net.")
    A(f"# FEATURES_V={features.FEATURES_V} NF={features.NF} "
      f"arch={wmeta['arch']}")
    A(f"# seeds: init={SEED_INIT} shuffle={SEED_SHUFFLE} "
      f"(shard order + within-shard permutation, per epoch)")
    A(f"# epochs_max={epochs} lr={args.lr} batch_size={args.batch_size} "
      f"patience={args.patience} optimizer=Adam loss=MSE(4 rotated outputs) "
      f"ridge_lambda={RIDGE_LAMBDA} (intercept unpenalised)")
    A(f"# shards: train={len(tr_paths)} val={len(va_paths)} "
      f"test={len(te_paths)}; test rows={teX.shape[0]} "
      f"test games={len(test_seeds)}")
    A("# BASELINES 3/4 computed on ALL test rows (no subsample needed: "
      f"{len(test_seeds)} game replays + {len(test_seeds) * 52} kernel "
      f"playouts took {playout_s:.1f}s)")
    A("")
    A("## device measurement (training steps/s, fwd+bwd+Adam)")
    for r in dev_results:
        A(f"  {r['device']:4s}  {r['rows_per_s']:12.0f} rows/s   "
          f"({r['steps']} steps x batch {r['batch_size']} in "
          f"{r['seconds']:.3f}s)")
    A(f"  chosen device: {device}"
      f"{' (forced via --device)' if args.device else ' (fastest measured)'}")
    A(f"  training wall time: {train_s:.1f}s; ridge fit {ridge_s:.1f}s")
    A("  Determinism: weight init and batch order are seeded. Residual "
      "library nondeterminism is NOT ruled out -- float reductions on MPS "
      "and multithreaded CPU BLAS are not guaranteed bit-reproducible, so "
      "byte-identical re-runs are the goal, statistically-identical the "
      "documented fallback (PHASE4_PLAN.md).")
    A("")
    A("## learning curve (MSE over all 4 rotated outputs)")
    A("  epoch    train_mse      val_mse   val_mse_seat0      rows")
    for i, ep in enumerate(hist["epoch"]):
        A(f"  {ep:5d} {hist['train_mse'][i]:12.4f} "
          f"{hist['val_mse'][i]:12.4f} {hist['val_mse_seat0'][i]:15.4f} "
          f"{hist['n_rows'][i]:9d}")
    A(f"  best epoch={hist['best_epoch']} best_val_mse="
      f"{hist['best_val_mse']:.4f} (weights restored to best epoch)")
    A(f"  net MSE on evaluated seat: val={train_mse_eval:.4f} "
      f"train-shard-0={net_train_mse:.4f} test={overall['MLP (value net)']:.4f}"
      f"  -> train-vs-test gap = "
      f"{overall['MLP (value net)'] - net_train_mse:+.4f}")
    A("")
    A("## TEST MSE on the evaluated seat's remaining points "
      "(shard column 0, verified)")
    A(f"  {'predictor':<58} {'overall':>9} {'heur-mix':>9} {'rand-mix':>9}")
    for k in order:
        A(f"  {k:<58} {overall[k]:9.4f} {heur_only[k]:9.4f} "
          f"{rand_only[k]:9.4f}")
    A(f"  (test rows: heuristic-mix={int(is_heur.sum())} "
      f"randomized-mix={int((~is_heur).sum())})")
    A("")
    A("## TEST MSE by trick bucket (ply//4+1; tricks 1-4 = plies 0-15, "
      "5-9 = 16-35, 10-13 = 36-51)")
    A(f"  {'predictor':<58}" + "".join(f"{n:>14}" for n, _, _ in BUCKETS))
    for k in order:
        A(f"  {k:<58}" + "".join(f"{buckets[k][n]:14.4f}"
                                 for n, _, _ in BUCKETS))
    A("  Early positions are genuinely harder (13 tricks of variance still "
      "to come); the buckets are reported so the overall average does not "
      "hide it.")
    A("")
    A("## npz export round-trip (torch float32 vs numpy float64 from the "
      f".npz), {n_chk} test rows")
    A(f"  max |numpy - torch| = {abs_err:.3e}")
    A(f"  max relative diff   = {rel_err:.3e}  (denominator floored at 1 "
      "point: the reference is torch float32, whose own representation "
      "error is ~6e-8 relative, and targets live on 0..26 points -- this "
      "measures agreement in POINTS)")
    A(f"  naive |diff|/|ref|  = {rel_err_naive:.3e}  (over |ref| > {floor}; "
      "dominated by float32 rounding on near-zero outputs)")
    A("")
    A("## PRE-REGISTERED SUCCESS CRITERIA (stated verbatim, not tuned)")
    A(f"  'MLP < linear < constant on test MSE': "
      f"{'PASS' if c1 else 'FAIL'}  "
      f"({overall['MLP (value net)']:.4f} < "
      f"{overall['linear (ridge)']:.4f} < "
      f"{overall['constant (per-ply mean)']:.4f})")
    A(f"  'MLP beats baseline 3 (one playout)': "
      f"{'PASS' if c2 else 'FAIL'}  "
      f"({overall['MLP (value net)']:.4f} vs "
      f"{overall['baseline 3: 1 heuristic playout']:.4f})")
    A(f"  [diagnostic, NOT the gate] same comparison on randomized-mix rows "
      f"only: {'MLP better' if c2_rand else 'playout better'} "
      f"({rand_only['MLP (value net)']:.4f} vs "
      f"{rand_only['baseline 3: 1 heuristic playout']:.4f})")
    A("")
    A("## RECORDED PLAN CORRECTIONS (read before acting on the gate)")
    A("  (a) BASELINE 4 IS DEGENERATE. Heuristic playouts are deterministic "
      "given a state, so 20 playouts from one position are 20 IDENTICAL "
      "playouts: baseline 4 is numerically identical to baseline 3 and is "
      "reported as such rather than faked with synthetic variance. The "
      "variance the plan was reaching for comes from re-sampling different "
      "determinized worlds from the belief table (a search world-batch) -- "
      "that needs a belief state, i.e. Task 5, not this script.")
    A("  (b) BASELINE 3 IS AN ORACLE ON THE HEURISTIC-MIX ROWS. Those games "
      "were played BY HeuristicPlayer and the kernel playout is a bitwise "
      "port of that same policy, so 'one playout from here' reproduces the "
      "recorded continuation card-for-card: MSE exactly 0. This script "
      "ASSERTS that exactness (it doubles as the correctness gate on "
      "position reconstruction). Because the split key (seed%100) and the "
      "mix key (seed%5) are not independent, TEST is 60% heuristic-mix, so "
      "the overall baseline-3 number is ~0.4x its randomized-mix value and "
      "measures data-generation coupling, not estimator quality. The gate "
      "above is still reported verbatim on the overall number; the "
      "randomized-mix column is the closest honest estimator comparison "
      "available here. If the gate FAILS, the lead/owner should weigh "
      "whether it failed for this artifact rather than on the merits.")
    A("  (b2) EVEN THE RANDOMIZED-MIX COLUMN FLATTERS BASELINE 3. Those "
      "games deviate from the heuristic only ~10% of plies "
      "(RandomizedHeuristic eps=0.1), so late positions leave few plies for "
      "the pure-heuristic playout to diverge on and the oracle effect comes "
      "back: see baseline 3's tricks-10-13 bucket. The rand-mix column is "
      "the honestEST comparison available, not a clean one.")
    A("  (c) Determinism observed empirically: two consecutive smoke runs on "
      "MPS produced a BYTE-IDENTICAL exported .npz (sha256 equal). That is "
      "evidence, not a guarantee, at full scale.")
    txt = "\n".join(lines) + "\n"
    txt_path, png_path = TXT_PATH, PNG_PATH
    if args.tag != "v1":
        txt_path = os.path.join(RESULTS, f"value_train_{args.tag}.txt")
        png_path = os.path.join(RESULTS, f"value_train_{args.tag}.png")
    with open(txt_path, "w") as f:
        f.write(txt)
    with open(os.path.join(OUT_DIR, "history.json"), "w") as f:
        json.dump({"history": hist, "overall": overall, "buckets": buckets,
                   "heur_only": heur_only, "rand_only": rand_only,
                   "devices": dev_results, "device": device}, f, indent=2)
    make_plot(png_path, hist, buckets, order[:4])
    print(txt)
    print(f"wrote {txt_path}, {png_path}, {ckpt}, {npz_path}")


if __name__ == "__main__":
    main()
