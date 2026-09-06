"""Train-only diagnostics for the Phase-7A expert population (plan 7A-1 §4 G5).

Reports, appended to results/expert_diag_v1.txt:
  * the pre-registered floors (pairwise divergence, vs-heuristic) verbatim;
  * per-rule FIRING rate (the rule contributed to some candidate) and
    ACTION-CHANGING rate (zeroing that weight changes the greedy choice),
    split by decision class and hand phase -- the effective-width instrument;
  * the deciding-stage histogram;
  * within-expert and between-expert entropy in nats;
  * Python ms/decision (the 7C cost basis; a report, not a claim).
TRAIN IDS ONLY.  Held-out ids are refused (the wall).
"""
import argparse
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from openhearts.engine import cards
from openhearts.engine.game import deal
from openhearts.players.expert import (
    ExpertPlayer, explain, greedy_pick, action_distribution, action_changing_rules,
)
from openhearts.players.expert_population import (
    OPTIONAL_WEIGHTS, REFERENCE_PARAMS, MASTER_SEED_EXPERT_V1, is_heldout,
    sample_expert, train_ids,
)
from openhearts.players.heuristic import HeuristicPlayer


def views_from_driver(seeds, params):
    out = []
    for s in seeds:
        st = deal(np.random.default_rng(s))
        drv = ExpertPlayer(np.random.default_rng(0), params)
        while not st.is_over():
            v = st.view_for(st.to_play)
            if bin(v.legal_moves).count("1") > 1:
                out.append(v)
            st.play(drv.choose(v))
    return out


def phase(v):
    return "early" if v.trick_number <= 3 else ("late" if v.trick_number >= 10 else "mid")


def kind(v):
    if not v.current_trick:
        return "lead"
    led = cards.suit(v.current_trick[0][1])
    return "follow" if v.hand & cards.SUIT_MASK[led] else "discard"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="*", default=None,
                    help="train ids (default: all 200)")
    ap.add_argument("--deals", type=int, default=50)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--rule-ids", type=int, default=40,
                    help="how many train experts get the per-rule scan")
    ap.add_argument("--out", default="results/expert_diag_v1.txt")
    args = ap.parse_args()
    ids = args.ids or train_ids()
    bad = [i for i in ids if is_heldout(i)]
    if bad:
        sys.exit(f"refusing held-out ids {bad}: train-only diagnostics")

    t0 = time.time()
    views = views_from_driver(range(70000, 70000 + args.deals), REFERENCE_PARAMS)
    hviews = views_from_driver(range(71000, 71006), REFERENCE_PARAMS)
    lines = []
    P = lines.append
    P("=" * 78)
    P(f"expert_diagnostics v1  {time.strftime('%Y-%m-%d %H:%M')}  master={MASTER_SEED_EXPERT_V1}")
    P(f"corpus: {len(views)} multi-legal views from the reference driver on deals "
      f"70000-{70000 + args.deals - 1}; {len(hviews)} on 71000-71005; train ids n={len(ids)}")

    # floors
    rng = np.random.default_rng(2026)
    rates = []
    for _ in range(args.pairs):
        a, b = rng.choice(ids, size=2, replace=False)
        pa, pb = sample_expert(int(a)), sample_expert(int(b))
        rates.append(np.mean([greedy_pick(v, pa) != greedy_pick(v, pb) for v in views]))
    P(f"[floor i] pairwise divergence over {args.pairs} pairs: mean={np.mean(rates):.3f} "
      f"min={min(rates):.3f} max={max(rates):.3f}  (floor >= 0.12; duplicates: "
      f"{sum(1 for r in rates if r == 0)})")
    h = HeuristicPlayer()
    hr = {i: np.mean([greedy_pick(v, sample_expert(i)) != h.choose(v) for v in views])
          for i in ids}
    P(f"[floor ii] vs plain heuristic: mean={np.mean(list(hr.values())):.3f} "
      f"min={min(hr.values()):.3f} (id {min(hr, key=hr.get)}) max={max(hr.values()):.3f} "
      f"(floors: each >= 0.10, mean >= 0.20)")
    rr = {i: np.mean([greedy_pick(v, sample_expert(i)) != greedy_pick(v, REFERENCE_PARAMS)
                      for v in views]) for i in ids}
    P(f"[report] vs reference config: mean={np.mean(list(rr.values())):.3f} "
      f"min={min(rr.values()):.3f} max={max(rr.values()):.3f}")

    # per-class pairwise disagreement
    cls = defaultdict(list)
    for _ in range(min(args.pairs, 40)):
        a, b = rng.choice(ids, size=2, replace=False)
        pa, pb = sample_expert(int(a)), sample_expert(int(b))
        per = defaultdict(list)
        for v in views:
            per[(kind(v), phase(v))].append(greedy_pick(v, pa) != greedy_pick(v, pb))
        for k, xs in per.items():
            cls[k].append(np.mean(xs))
    P("[report] pairwise disagreement by decision class x phase (40 pairs):")
    for k in sorted(cls):
        P(f"    {k[0]:8s} {k[1]:6s} {np.mean(cls[k]):.3f}  (n views {sum(1 for v in views if (kind(v), phase(v)) == k)})")

    # firing / action-changing / stage histogram
    scan_ids = ids[:args.rule_ids]
    fire = Counter(); change = Counter(); n_by_kind = Counter(); stages = Counter()
    fire_k = defaultdict(Counter); change_k = defaultdict(Counter)
    for i in scan_ids:
        p = sample_expert(i)
        for v in views:
            k = kind(v)
            n_by_kind[k] += 1
            ex = explain(v, p)
            stages[ex["returned_at"]] += 1
            for r in ex["fired"]:
                fire[r] += 1; fire_k[k][r] += 1
            for w in action_changing_rules(v, p):
                change[w] += 1; change_k[k][w] += 1
    tot = sum(n_by_kind.values())
    P(f"[report] deciding stage histogram over {len(scan_ids)} experts x {len(views)} views:")
    for st, c in sorted(stages.items(), key=lambda x: -x[1]):
        P(f"    {st:6s} {c / tot:.3f}")
    P("[report] rule FIRING rate (rule contributed to some candidate), overall and by class:")
    for r in sorted(fire):
        P(f"    {r:8s} {fire[r] / tot:.3f}   "
          + "  ".join(f"{k}={fire_k[k][r] / max(1, n_by_kind[k]):.3f}"
                      for k in ("lead", "follow", "discard")))
    P("[report] weight ACTION-CHANGING rate (zeroing that one weight changes the greedy "
      "choice), overall and by class:")
    for w in OPTIONAL_WEIGHTS:
        P(f"    {w:12s} {change[w] / tot:.3f}   "
          + "  ".join(f"{k}={change_k[k][w] / max(1, n_by_kind[k]):.3f}"
                      for k in ("lead", "follow", "discard")))
    eff = sum(1 for w in OPTIONAL_WEIGHTS if change[w] / tot >= 0.005)
    P(f"[report] optional weights that change >= 0.5% of decisions: {eff} of {len(OPTIONAL_WEIGHTS)}")

    # entropy
    within = defaultdict(list); mix = []; contested = []
    for v in views[::4]:
        agg = defaultdict(float)
        for i in ids:
            p = sample_expert(i)
            cs, ps = action_distribution(v, p)
            within[p.tie_mode].append(-sum(x * math.log(x) for x in ps if x > 0))
            for c, x in zip(cs, ps):
                agg[c] += x / len(ids)
        hm = -sum(x * math.log(x) for x in agg.values() if x > 0)
        mix.append(hm)
        if len(agg) > 1:
            contested.append(hm)
    P(f"[report] entropy (nats): within fixed={np.mean(within['fixed']):.4f}, "
      f"within seeded_uniform={np.mean(within['seeded_uniform']):.4f}; between-expert "
      f"mixture={np.mean(mix):.3f} over {len(mix)} views, {np.mean(contested):.3f} on "
      f"{len(contested)} contested ({len(contested) / len(mix):.1%})")

    # cost
    p = sample_expert(ids[0])
    t1 = time.time()
    for v in views:
        greedy_pick(v, p)
    P(f"[report] python reference cost: {(time.time() - t1) / len(views) * 1000:.3f} ms/decision "
      f"(greedy, one train expert, this corpus, single process)")
    P(f"wall {time.time() - t0:.1f} s")
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
