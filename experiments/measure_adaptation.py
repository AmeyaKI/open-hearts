"""Phase 5 Task 5 measurement: how much of the adaptation headroom does a
K-member mixture actually capture, on personalities it has never seen?

WHAT IS MEASURED, and the protocol that makes it honest
-------------------------------------------------------
For each target personality we play H hands at a table of its own kind (the
target's three tablemates are drawn from the same pool as the target, so the
seat we read is embedded in a normal game, not a lab fixture).  Then, for
h = 1 .. H-1:

    fit the seat's `SeatMixture` on hands 1..h
    score hand h+1's decision events            <- OUT OF SAMPLE, always

Scoring a hand with weights that already saw it would make the gate pass for
free, so the evaluation hand is never in the fit.  Five scorers on the SAME
event set:

  GENERIC     the deployment model for unseen opponents (constant in h)
  MIXTURE     sum_k w_k P_k, weights from hands 1..h
  MIX+BLEND   (1-b)*MIXTURE + b*GENERIC, b = adapt.BLEND_B
  COND-TRUE   the CONDITIONED net with the target's TRUE parameters --
              deployment-impossible; this is Task 3's headroom ceiling
              (+0.1136 nats / +8.17 top-1 over GENERIC on its own eval set)
  MAP         the single heaviest member of the fitted mixture, scored
              alone.  Out-of-sample and deployment-realizable; read against
              MIXTURE it says whether averaging or committing is better.
  BEST-POOL   the single best pool member CHOSEN IN HINDSIGHT on the
              evaluation hand.  NOT a method and NOT a valid bound: it is a
              max over K candidates scored on the same ~40 events that chose
              it, so it is UPWARD BIASED by selection and can (and in the
              smoke run does) exceed COND-TRUE.  Read it only as an
              optimistic ceiling on pool coverage, never as evidence that a
              concentrated mixture could reach it.

Headroom captured = (MIXTURE - GENERIC) / (COND-TRUE - GENERIC), reported at
hand 5 and hand 10 (i.e. having observed 5 and 10 hands).

MODES
-----
  --sweep   the PRE-REGISTERED blend sweep, TRAIN-side only (personalities in
            neither the pool nor the held-out set, seeds from
            adapt.SWEEP_SEED_BASE).  Run BEFORE any held-out number; its
            result is what `adapt.BLEND_B` records.
  --smoke   small held-out run (8 personalities x 4 hands), end-to-end check
  --full    the reported run: >=40 held-out personalities x >=10 hands

Features come from `gen_population_data.play_and_record`, the frozen Task-2
extractor, so the rows are byte-identical to the profiler's training diet
(float16 there, widened to float64 here -- identically for every scorer, so no
arm is advantaged).

* = upward-biased hindsight selector, see BEST-POOL above.

Seeds: held-out evaluation games use base 980000 (distinct from Task 3's
held-out eval base 950000 and from the training range 700000+).
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import gen_population_data as G  # noqa: E402
from openhearts.opponent.adapt import (BLEND_GRID, DEFAULT_K,  # noqa: E402
                                       SWEEP_SEED_BASE, PoolProfiler,
                                       SeatMixture, pool_ids)
from openhearts.opponent.infer import load_profiler  # noqa: E402
from openhearts.opponent.params import PARAM_DIM, param_vector  # noqa: E402
from openhearts.engine.features import NF  # noqa: E402
from openhearts.opponent.infer import profiler_probs_batch  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
COND_NPZ = os.path.join(RESULTS, "profiler_train", "profiler_conditioned.npz")
GENERIC_NPZ = os.path.join(os.path.dirname(__file__), "..", "models",
                           "profiler_v1.npz")
EVAL_SEED_BASE = 980_000


# ------------------------------------------------------------------ data
def match_events(target_pid, tablemates, seed_base, n_hands):
    """Play `n_hands` hands with `target_pid` seated; per-hand target events.

    Returns a list of `(feats float64[N,NF], masks int64[N], chosen int64[N])`,
    one entry per hand, covering only the plies where the TARGET acted.
    """
    pool = [int(target_pid)] + [int(x) for x in tablemates]
    assert len(set(pool)) == 4
    out = []
    for i in range(n_hands):
        seed = int(seed_base) + i
        rows, table = G.play_and_record(seed, pool=pool)
        seat = table.index(int(target_pid))
        sel = np.asarray(rows["acting_seat"]) == seat
        out.append((rows["profiler_features"][sel].astype(np.float64),
                    rows["legal_mask"][sel],
                    rows["chosen_card"][sel].astype(np.int64)))
    return out


def single_probs(weights, feats, masks, params=None):
    """[N,52] for one parameter vector (or the GENERIC net when None)."""
    feats = np.asarray(feats, dtype=np.float64)
    n = feats.shape[0]
    if n == 0:
        return np.zeros((0, 52))
    if params is None:
        big = np.ascontiguousarray(feats)
    else:
        big = np.empty((n, NF + PARAM_DIM), dtype=np.float64)
        big[:, :NF] = feats
        big[:, NF:] = params
    out = np.zeros((n, 52), dtype=np.float64)
    profiler_probs_batch(*weights, big, np.asarray(masks, dtype=np.int64), out)
    return out


def score(P, chosen):
    """(sum log p, n, n top-1 correct) for a [N,52] probability block."""
    if P.shape[0] == 0:
        return 0.0, 0, 0
    p = P[np.arange(P.shape[0]), chosen]
    assert (p > 0).all(), "truth-safety violated: zero on a chosen legal card"
    return float(np.log(p).sum()), int(P.shape[0]), int(
        (P.argmax(axis=1) == chosen).sum())


# ------------------------------------------------------------- the sweep
def run_arm(targets, tablemates_for, seed_base_for, n_hands, pool, cond_w,
            gen_w, blends, log=print):
    """Trajectory accumulators over `targets`.  Returns a dict of arrays.

    Index h (0-based) means "having observed h+1 hands, scored on hand h+2".
    """
    H = n_hands - 1
    acc = {name: np.zeros((H, 3)) for name in
           ["GENERIC", "MIXTURE", "MAP", "COND-TRUE", "BEST-POOL"]}
    for b in blends:
        acc[f"BLEND{b}"] = np.zeros((H, 3))
    t_update = [0.0, 0]
    t_call = [0.0, 0]

    for ti, pid in enumerate(targets):
        hands = match_events(pid, tablemates_for(pid), seed_base_for(pid),
                             n_hands)
        mix = SeatMixture(pool)
        true_p = param_vector(int(pid))
        for h in range(H):
            f, m, c = hands[h]
            t0 = time.perf_counter()
            mix.observe(f, m, c)
            t_update[0] += time.perf_counter() - t0
            t_update[1] += len(c)

            ef, em, ec = hands[h + 1]
            if len(ec) == 0:
                continue
            w = mix.weights
            t0 = time.perf_counter()
            pm = pool.member_probs(ef, em)              # [N,K,52]
            t_call[0] += time.perf_counter() - t0
            t_call[1] += len(ec)
            P_mix = np.einsum("k,nkc->nc", w, pm)
            P_gen = single_probs(gen_w, ef, em)
            P_true = single_probs(cond_w, ef, em, true_p)

            acc["GENERIC"][h] += score(P_gen, ec)
            acc["MIXTURE"][h] += score(P_mix, ec)
            acc["MAP"][h] += score(pm[:, int(w.argmax()), :], ec)
            acc["COND-TRUE"][h] += score(P_true, ec)
            for b in blends:
                acc[f"BLEND{b}"][h] += score(
                    (1 - b) * P_mix + b * P_gen, ec)
            # hindsight best single member on THIS hand
            lp = np.log(pm[np.arange(len(ec)), :, ec]).sum(axis=0)
            k = int(lp.argmax())
            acc["BEST-POOL"][h] += score(pm[:, k, :], ec)
        if (ti + 1) % 10 == 0:
            log(f"  ... {ti + 1}/{len(targets)} targets", flush=True)
    return acc, t_update, t_call


def summarize(acc):
    """{name: (mean LL per event, top-1 %) arrays over h}."""
    out = {}
    for name, a in acc.items():
        n = np.maximum(a[:, 1], 1)
        out[name] = (a[:, 0] / n, 100.0 * a[:, 2] / n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    args = ap.parse_args()
    if not (args.smoke or args.full or args.sweep):
        ap.error("pass --smoke, --full or --sweep")

    assert os.path.exists(COND_NPZ), (
        f"{COND_NPZ} missing: rerun experiments/train_profiler.py "
        "(gitignored, regenerable)")
    cond_w, cond_meta = load_profiler(COND_NPZ)
    gen_w, _gm = load_profiler(GENERIC_NPZ)
    ids = pool_ids(args.k)
    pool = PoolProfiler(cond_w, ids)

    train_ids, heldout_ids = G.TRAIN_IDS, G.HELDOUT_IDS
    lines = []

    def log(*a, **kw):
        msg = " ".join(str(x) for x in a)
        print(msg, **kw)
        lines.append(msg)

    log(f"# Phase 5 Task 5 -- mixture adaptation, K={pool.k}")
    log(f"# pool ids (TRAIN, seeded, salt {5150}): "
        f"{','.join(str(i) for i in ids)}")
    log(f"# conditioned net: {COND_NPZ} arch={cond_meta.get('arch')}")

    if args.sweep:
        # PRE-REGISTERED, TRAIN-ONLY. Targets: train personalities NOT in the
        # pool; tablemates likewise. Never touches held-out ids.
        cand = [i for i in train_ids if i not in set(ids)]
        rng = np.random.default_rng([314159, 5151])
        targets = [int(cand[int(i)])
                   for i in rng.choice(len(cand), size=40, replace=False)]
        rest = [i for i in cand if i not in set(targets)]

        def mates(pid, rest=rest):
            r = np.random.default_rng([int(pid), 5152])
            return [int(rest[int(i)])
                    for i in r.choice(len(rest), size=3, replace=False)]

        n_hands = 5   # fit on 1..h, score h+1; h=3 is the pre-registered row
        log(f"\n## BLEND SWEEP (TRAIN-side only, {len(targets)} personalities "
            f"x {n_hands} hands, seeds {SWEEP_SEED_BASE}+)")
        acc, tu, tc = run_arm(
            targets, mates, lambda pid: SWEEP_SEED_BASE + 100 * (
                targets.index(pid)), n_hands, pool, cond_w, gen_w,
            BLEND_GRID, log=log)
        s = summarize(acc)
        log(f"{'hands_obs':>9} " + " ".join(f"{n:>12}" for n in
                                            ["GENERIC", "MIXTURE"] +
                                            [f"b={b}" for b in BLEND_GRID]))
        for h in range(n_hands - 1):
            log(f"{h + 1:>9} " + " ".join(
                f"{v:>12.4f}" for v in
                [s['GENERIC'][0][h], s['MIXTURE'][0][h]] +
                [s[f'BLEND{b}'][0][h] for b in BLEND_GRID]))
        row = 2   # h index for "fitted on hands 1-3, scored on hand 4"
        vals = [(s[f"BLEND{b}"][0][row], b) for b in BLEND_GRID]
        best = max(vals, key=lambda t: (round(t[0], 6), -t[1]))
        log(f"\nSELECTION (fit hands 1-3, score hand 4): " +
            ", ".join(f"b={b}: {v:.4f}" for v, b in vals))
        log(f"CHOSEN b = {best[1]} (LL {best[0]:.4f}); "
            "ties broken toward the smaller b")
        out = os.path.join(RESULTS, "adaptation_sweep.txt")
        with open(out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        log(f"wrote {out}")
        return

    # ---------------------------------------------------------- held-out
    from openhearts.opponent.adapt import BLEND_B
    n_targets, n_hands = (8, 4) if args.smoke else (40, 10)
    n_hands = n_hands + 1   # need an extra hand to score the last fit on
    targets = [int(x) for x in heldout_ids[:n_targets]]
    rest = [int(x) for x in heldout_ids]

    def mates(pid, rest=rest):
        r = np.random.default_rng([int(pid), 5153])
        cand = [i for i in rest if i != pid]
        return [int(cand[int(i)])
                for i in r.choice(len(cand), size=3, replace=False)]

    log(f"\n## HELD-OUT ({n_targets} personalities x {n_hands} hands, seeds "
        f"{EVAL_SEED_BASE}+, blend b={BLEND_B})")
    assert set(targets).isdisjoint(set(ids)), "held-out wall violated"
    t0 = time.perf_counter()
    acc, tu, tc = run_arm(
        targets, mates,
        lambda pid: EVAL_SEED_BASE + 100 * targets.index(pid),
        n_hands, pool, cond_w, gen_w, (BLEND_B,), log=log)
    elapsed = time.perf_counter() - t0
    names = ["GENERIC", "MIXTURE", f"BLEND{BLEND_B}", "MAP", "COND-TRUE",
             "BEST-POOL*"]
    acc["BEST-POOL*"] = acc.pop("BEST-POOL")
    s = summarize(acc)
    log("\nmean log-likelihood per decision event (out-of-sample: fitted on "
        "hands 1..h, scored on hand h+1)")
    log(f"{'hands_obs':>9} " + " ".join(f"{n:>11}" for n in names))
    for h in range(n_hands - 1):
        log(f"{h + 1:>9} " + " ".join(f"{s[n][0][h]:>11.4f}" for n in names))
    log("\ntop-1 %")
    log(f"{'hands_obs':>9} " + " ".join(f"{n:>11}" for n in names))
    for h in range(n_hands - 1):
        log(f"{h + 1:>9} " + " ".join(f"{s[n][1][h]:>11.2f}" for n in names))

    g = s["GENERIC"][0]
    head = s["COND-TRUE"][0] - g
    # GENERIC is NOT constant down this column: each row h is scored on a
    # DIFFERENT hand (h+1), so hand-to-hand difficulty moves every arm
    # together. The paired DELTA below is the signal; the levels above are
    # context.
    log("\npaired delta vs GENERIC on the SAME events (nats)")
    log(f"{'hands_obs':>9} " + " ".join(f"{n:>11}" for n in names[1:]))
    for h in range(n_hands - 1):
        log(f"{h + 1:>9} " + " ".join(
            f"{s[n][0][h] - g[h]:>+11.4f}" for n in names[1:]))
    log("\nheadroom captured = (MIXTURE - GENERIC) / (COND-TRUE - GENERIC)")
    for h in range(n_hands - 1):
        frac = (s["MIXTURE"][0][h] - g[h]) / head[h] if head[h] > 0 else \
            float("nan")
        fb = (s[f"BLEND{BLEND_B}"][0][h] - g[h]) / head[h] if head[h] > 0 \
            else float("nan")
        log(f"  after {h + 1:>2} hands: headroom {head[h]:+.4f} nats; "
            f"MIXTURE {100 * frac:6.1f}%  BLEND {100 * fb:6.1f}%")

    per_up = 1e6 * tu[0] / max(tu[1], 1)
    per_call = 1e6 * tc[0] / max(tc[1], 1)
    log(f"\ncost: mixture update {per_up:.1f} us/observed event "
        f"({tu[1]} events); adapted likelihood {per_call:.1f} us/ply "
        f"(K={pool.k} member forward passes each); wall {elapsed:.1f}s")
    log(f"Task-6 projection: an audit of 100 worlds x ~29 opponent plies is "
        f"~{100 * 29 * per_call / 1e6:.2f}s per posterior at K={pool.k}. "
        f"The ratio is the number that matters: the RIA row costs ~"
        f"{per_call / 12.0:.0f}x the profiled-R row's SEARCH wall time (a "
        "GENERIC ply measures 12 us on this machine). Task 4's 500-game "
        "guessing run took ~13 min for 5 curves at 8 workers; an RIA row of "
        "that size projects to hours, so either budget it, cut its deal "
        "count, or prune the pool to the top-M members (a one-line change in "
        "PoolProfiler.member_probs -- but only with a profile in hand).")

    tag = "_smoke" if args.smoke else ""
    out = os.path.join(RESULTS, f"adaptation{tag}.txt")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"wrote {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(1, n_hands)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for n in names:
            ax[0].plot(x, s[n][0], marker="o", label=n)
            ax[1].plot(x, s[n][1], marker="o", label=n)
        ax[0].set_xlabel("hands observed")
        ax[0].set_ylabel("mean LL / event (next hand)")
        ax[0].set_title(f"Adaptation on held-out personalities (K={pool.k})")
        ax[1].set_xlabel("hands observed")
        ax[1].set_ylabel("top-1 %")
        ax[0].legend(fontsize=7)
        ax[0].grid(alpha=.3)
        ax[1].grid(alpha=.3)
        fig.tight_layout()
        png = os.path.join(RESULTS, f"adaptation{tag}.png")
        fig.savefig(png, dpi=130)
        print(f"wrote {png}")
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
