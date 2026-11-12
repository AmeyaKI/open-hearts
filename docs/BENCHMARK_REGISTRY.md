# Benchmark registry

This file lists every external or synthetic opponent that any open-hearts claim rests on, with exactly what it is and how far its numbers can be trusted; it has been maintained since 2026-09-26, and gaps are written as gaps. **Rule: no external claim leaves the repo unless its opponent has an entry here, and the claim uses that entry's canonical wording.** Unless stated otherwise, every entry uses **no-pass / no-moon Hearts** (`hearts(pass_cards=False, qs_breaks_hearts=False)` in OpenSpiel terms; standard per-trick scoring, Q♠ = 13, 2♣ leads trick 1, no shoot-the-moon rescoring). Points are per seat-hand, lower is better, and a 4-seat hand sums to 26.

Last reviewed: 2026-09-29.

## Summary

"Ours" is the shipped bot, TallyHearts (honest-FULL 50×20 fused). Headline results are points per hand (lower is better) unless stated.

| Opponent | What it is | Evidence class | Headline result | Replayable? |
|---|---|---|---|---|
| [Phase-1 heuristic](#phase-1-heuristic) | Hand-written beginner rules, deterministic given the view | Our own gym | ours 3.253 vs 3 heuristics | Yes |
| [Phase-5/6 personalities](#phase-56-personalities) | Parameterised heuristic variants, held-out trios | Our own gym (soft) | ours 2.345 vs held-out trios | Yes |
| [Expert population v1](#expert-population-v1) | 250 rulebook experts, 50 held out | Our own gym (ISMCTS-hard) | ours 4.761 vs held-out experts | Yes |
| [OpenSpiel generic ISMCTS](#openspiel-generic-ismcts) | Stock, untuned generic ISMCTS, 1,000 simulations | Our own measurement | 4.70 vs 7.10 | Yes, since 2026-09-26 |
| [OpenSpiel ISMCTS, domain-aware rungs](#openspiel-ismcts-domain-aware-rungs) | OpenSpiel ISMCTS with our heuristic rollout and prior; built by us | Our own, pre-registered | margin 1.10 (4× compute) / 1.57 (equal) | Yes |
| [xinxin](#xinxin-sturtevant) | Sturtevant's reference UCT Hearts program | Published program; our measurement | margin 0.776 per seat-hand | Our seats only |
| [GO-MCTS](#go-mcts-paper) | Full-rules Hearts agent from a paper; no code | Published numbers | not run | No |
| [Berns Deep-CFR, Valet/CardStock](#berns-deep-cfr-and-valetcardstock) | Not acquired | Self-reported / framework | not run | No |

## Hardware and environments (all measurements)

| | |
|---|---|
| Machine | one Apple M5 Max (18 cores, 128 GB), macOS 26.6.2 — the only valid bench: x86 fails the project's bitwise tripwires by 1 ULP (ci-probe branch) |
| Project venv | Python 3.12.4, numpy 2.4.6, numba 0.66.0 (fused kernel bitwise-gated against the Python path) |
| pyspiel venv (C-bench, xinxin) | Python 3.12.12, numpy 2.5.3, open_spiel 2.0.2 built from source (google-deepmind/open_spiel @ 4840189, 2026-08-31) with `OPEN_SPIEL_BUILD_WITH_XINXIN=ON`; recipe `scripts/build_xinxin.sh`; rebuilt per session (scratch venvs do not persist) |
| Seeds | Deals: `seed_base + i` (100000+ for banked rows; probes/smokes ≥ 900000). Our seat rngs since 2026-09-26 (B0): `openhearts.eval.seeds.bench_seat_seed` (SplitMix64 `derive_seed`, domain 6, harness id 1 = C-bench, 2 = xinxin; header token `seeds=stable-v1`). **Rows banked before 2026-09-26 used Python's per-process-salted `hash()` for seat seeds: honest samples, not game-replayable.** Internal harnesses (ablation5/7/7c, exploit) use `numpy.default_rng([config_id, block])` or `derive_seed`, which are stable. |
| Statistics | Paired per-deal differences on identical deal seeds, 4 rotations (or 14 seatings), percentile bootstrap 95% CI over deals (`openhearts.eval.stats.bootstrap_ci`); Wilcoxon signed-rank reported where the literature uses it. Deal pairing removes deal luck; it does NOT pair search dice across bots (each bot has its own seat stream). |

## TallyHearts versions

The product name **TallyHearts** (2026-11-06) refers to the shipped honest-FULL 50×20 fused bot below; code and row names are unchanged.

| Version | Settings | Cost (s/decision, 12-worker contended unless noted) | Status |
|---|---|---|---|
| **honest-FULL 50×20 fused** (shipped) | exact belief table (Level.FULL), 50 outer worlds × 20 inner re-determinizations, fused kernel (bitwise-gated), grouping off, no posterior reading | 0.026–0.027 | shipped since Phase 6; exploitability +0.198 (+0.068, +0.327) champion-side, minor leak |
| honest-FULL unfused | same, Python path | 0.145 (C-bench 1 protocol ran it unfused) | bitwise identical to fused; historical |
| 200×20 / 100×50 fused | 7D-V0 candidates | 0.105 / 0.115 (heuristic field) | **not adopted** (2026-09-26): ~0.15 pts/hand better on every field, significant only vs heuristics |
| expert-rollout 50×20 | train experts as opponent playout policy (7C) | 0.22–0.25 | retired: null on all four fields, loses to equal-time control |
| ISMCTS-in-our-harness | — | — | none; we never ship a tree searcher |

## Opponents

### Phase-1 heuristic

- **What it is:** hand-written beginner rules (`src/openhearts/players/heuristic.py`), deterministic given the view.
- **Source and license:** this repo.
- **Settings:** none.
- **Seeds and replay:** deterministic; replayable.
- **Evidence class:** our own gym; no external calibration.
- **Result:** honest-FULL 3.253 pts/hand vs 3 heuristics (Phase 2; sweep 50×20 = 3.260 on 500 deals, 2026-09-26).
- **Compositions and tests:** 1v3 rotated; `tests/test_heuristic*.py`.

### Phase-5/6 personalities

- **What it is:** parameterised heuristic variants.
  - Populations v1 (master 314159, 200/50) and v2 (master 606060, `personality.py`).
  - Held-out trios from `run_ablation5.py` (`TRIO_SEED=[314159, 6060]`).
- **Source and license:** this repo.
- **Settings:** 20 held-out trios × deals 100000+.
- **Seeds and replay:** `numpy.default_rng` streams; replayable.
- **Evidence class:** our own gym, **soft** (Phase 6 showed a stock ISMCTS bot is harder).
- **Result:** honest-FULL 2.345 vs held-out trios (ablation5, 500 deals).
- **Compositions and tests:** 1v3 rotated; `tests/test_personality*.py`.

### Expert population v1

- **What it is:** 250 experts drawn from the project's rulebook of expert archetypes.
  - 32-field records, SplitMix from master 7,070,707.
  - Ids 1–200 are for training, 201–250 are held out; held-out parameters are never inspected during development.
- **Source and license:** this repo (`expert_population.py`, `expert.py`), corpus pin `3e73a45e…ee80`.
- **Settings:** 20 held-out trios (7B blocks), deals 100000+.
- **Seeds and replay:** `derive_seed`; replayable.
- **Evidence class:** our own gym, **ISMCTS-hard** (4.761 ≈ C-bench 4.769 on 500 deals).
- **Result:** honest-FULL **4.761 (4.553, 4.972)** vs held-out experts (7B, 500 deals).
- **Compositions and tests:** 1v3 rotated; `tests/test_expert.py` (56), `tests/test_expert_population.py` (15).

### OpenSpiel generic ISMCTS

- **What it is:** `open_spiel.python.algorithms.ismcts.ISMCTSBot`, `uct_c=2.0`, `max_simulations=1000`, `RandomRolloutEvaluator(n_rollouts=1)`; stock, untuned, no domain knowledge. **A generic baseline, never "frontier" or "a model".**
- **Source and license:** google-deepmind/open_spiel @ 4840189 (Apache-2.0) via `experiments/run_cbench.py`; adapter `experiments/cbench_adapter` (see `run_cbench.py` imports).
- **Settings:** 1,000 sims; also 3,000 / 10,000 (A4 curve).
- **Seeds and replay:** fully seeded (`np.random.RandomState(seed)`, same seat seed as ours). **Game-replayable since 2026-09-26** (run 1: two 12-worker processes byte-identical on 20 deals).
- **Evidence class:** our own measurement; OpenSpiel publishes no Hearts strength figures.
- **Result (canonical wording):**

  > Canonical: *"On 1,000 matched no-pass/no-moon deals, we beat OpenSpiel's stock, untuned generic ISMCTS baseline configured with 1,000 simulations and one random rollout (4.70 vs 7.10 points/hand; lower is better)."*

  - Majority: 5.77 vs 8.70.
  - Compute-robust: ISMCTS@1k/3k/10k → ours 5.02/5.18/5.19, n.s. (A4).
  - **Stable-seed replication** (ours-minority, 500 deals, seeds 100000+, 2026-09-26/27, `results/cbench_ours-minority_stableseed.txt`): ours 4.901, ISMCTS 7.033; replication 4.901 vs banked 4.769 on n=500 deals; replication − banked = +0.132 (-0.139, +0.405), within the pre-registered 0 ± 0.3 and n.s. **This replication is the canonical, replayable reference for the 1v3 C-bench figure.**
- **Compositions and tests:** ours-minority (1v3), ours-majority (3v1), rotated; `tests/test_cbench.py`.

### OpenSpiel ISMCTS, domain-aware rungs

- **What it is:** a ladder of stronger ISMCTS opponents (B1), **built by us from OpenSpiel's algorithm**: no Hearts-tuned ISMCTS player exists publicly (verified 2026-09-27). OpenSpiel's `ISMCTSBot` is unchanged, with `experiments/cbench/evaluators.py::HeuristicRolloutEvaluator` in its two hooks:
  - rung 1 = random rollout with held-out-tuned knobs;
  - rung 2 = our Phase-1 heuristic as the rollout;
  - rung 3 = rollout + heuristic move prior (ε=0.25) with PUCT;
  - rung 3e = rung 3 at equal compute to our shipped bot.
- **Source and license:** google-deepmind/open_spiel @ 4840189 (Apache-2.0) + this repo @ 5ce4828.
- **Settings:** knobs selected by coordinate descent on tuning deals 300000–300099 (never evaluated on): UCT c=0.5, unlimited world samples, MAX_VISIT_COUNT; sims 1000 / 2000 / 1200 / 330 (equal wall-clock probes at 12 workers).
- **Seeds and replay:** fully seeded (`seeds=stable-v1`); game-replayable. The same config re-run reproduced 100 deals identically.
- **Evidence class:** our own measurement, pre-registered before the runs, with stakes recorded before the runs.
- **Result (canonical wording):**

  > Canonical: *"On 500 matched no-pass/no-moon deals, open-hearts (50×20) outscored OpenSpiel's ISMCTS equipped with open-hearts' own card-play heuristic as its rollout policy and move prior by 1.10 points per hand (95% CI 0.81–1.40) when ISMCTS was given about four times our per-decision compute, and by 1.57 (1.29–1.86) at equal compute; the opponent was built by us from OpenSpiel's algorithm, its knobs selected on held-out deals."*

  Ladder (ours / ISMCTS):

  | Rung | Ours | ISMCTS |
  |---|---|---|
  | stock | 4.901 | 7.033 |
  | tuned (tuning made it worse: +0.144 (0.051, 0.239)) | 4.469 | 7.177 |
  | heuristic rollout | 5.696 | 6.768 |
  | + prior | 5.672 | 6.776 |
  | equal compute | 5.323 | 6.892 |
- **Caveats:**
  - **Disclosed baseline flaws inherited from OpenSpiel:** moon-rule returns in the stock/tuned rungs; a resampler that ignores discard-implied voids in every rung (≈38% of evaluated worlds impossible).
  - Rungs 2–3 ran at c=0.5; c=2.0 variants not yet run.
- **Compositions and tests:** ours-minority (1v3) rotated only; `tests/test_b1_evaluators.py` (9, pyspiel venv), including byte-identity of the stock rung at defaults.

### xinxin (Sturtevant)

- **What it is:** the field's reference Hearts program (UCT-based, 25 years), via OpenSpiel's `bots/xinxin` wrapper.
  - Plays no-pass natively.
  - No-moon via its **native `kNoShooting` flag**, enabled by `experiments/xinxin_knoshooting.patch` (20 lines; rules config, not a strength change; disclosed in every header).
- **Source and license:** nathansttt/hearts @ 6ba1154 (2020-08-14), **as cloned at HEAD by OpenSpiel's `install.sh`; not pinned by us until 2026-09-26 (B0), when the commit was recorded**. GPL-3.0 (LICENSE in the checkout); wrapper Apache-2.0.
- **Settings:** paper config: 2,000 runs, 50 worlds, C = 0.4, threads off; Release build.
- **Seeds and replay:** **uncontrollable**: the wrapper exposes no seed knob. Our seats are stable since 2026-09-26, but games are NOT replayable. Between-run noise measured **−0.050 (−0.304, +0.197)** on 500 deals (2026-09-25 rerun vs banked).
- **Evidence class:** published program (Sturtevant). The GO-MCTS paper reports beating it by 1.74 pts/hand WITH passing+moon: method-comparable only, never number-comparable. Our measurement is independent.
- **Result (canonical wording):**

  > Canonical: *"On 1,000 randomly dealt no-pass/no-moon Hearts hands across all 14 mixed seating configurations, our 50×20 fused bot beat xinxin at its published 2,000-run/50-world/C=0.4 configuration by 0.776 points per seat-hand (paired bootstrap 95% CI [0.642, 0.910] in our favor)"*

  Never "we beat xinxin at Hearts".
  - Tier 1: minority −0.809 [−1.100, −0.509], majority −0.555 [−0.879, −0.231].
  - Speed: ours 0.027 vs xinxin ~0.25 s/decision → **"~9× faster"** (shipped bot only).
  - **Stable-seed replication** (tier 1 ours-minority, 500 deals, seeds 100000+, 2026-09-27, `results/xinxin_tier1_ours-minority_stableseed.txt`):
    - ours 5.978 [5.746, 6.213], xinxin 6.674 [6.596, 6.751], ours − xinxin −0.696 [−1.005, −0.383], Wilcoxon z = −4.85;
    - vs banked (Aug 29): 5.978 vs 5.894, n=500, diff +0.085 (-0.165, +0.327);
    - vs 9/25 rerun: 5.978 vs 5.844, n=500, diff +0.134 (-0.116, +0.388);
    - within the pre-registered 0 ± 0.3 and n.s.
  - **Three same-deal runs of the same bot now span 5.84–5.98: that spread (our dice + xinxin's unseeded dice) is the run-to-run noise any single xinxin result carries. This replication is the canonical reference for the 1v3 tier-1 figure (our seats replayable; xinxin's are not).**
- **Compositions and tests:**
  - Tier 1: 1v3 and 3v1 rotated.
  - Tier 2: 14-way duplicate (4×1v3, 6×2v2, 4×3v1).
  - `run_xinxin_match.py --selftest` (Wilcoxon); rescore identity asserted every game (seats sum to 26); zero tripwires in 18,000+ games.

### GO-MCTS (paper)

- **What it is:** a full-rules Hearts agent, 25.6 s/move.
- **Source and license:** no code or checkpoint available.
- **Settings:** —
- **Seeds and replay:** —
- **Evidence class:** published numbers, not reproducible by us.
- **Result:** method-comparable only.
- **Compositions and tests:** —

### Berns Deep-CFR and Valet/CardStock

- **What it is:** not acquired; no measurement.
- **Source and license:** not acquired.
- **Settings:** —
- **Seeds and replay:** —
- **Evidence class:** self-reported / framework.
- **Result:** none.
- **Compositions and tests:** —

## Reproducibility status by banked row

| Banked row | Seat seeds | Replayable? | Note |
|---|---|---|---|
| C-bench ours-minority / ours-majority 1,000 deals (2026-08-2x) | salted `hash()` | no (statistically valid sample) | A3 env-check on a rebuilt venv: +0.170 (−0.233, +0.582) n.s. on 250 deals; the 2026-09-26/27 stable-seed replication supersedes it as the canonical reference |
| xinxin tier 1 / tier 2 (2026-08-29) | salted `hash()` on our seats; xinxin unseeded | no | 2026-09-25 same-deal rerun: −0.050 (−0.304, +0.197); run noise small |
| ablation5 / ablation7 / ablation7c / exploit rows | `numpy.default_rng([config, block])` / `derive_seed` | yes (block-level) | fused kernel bitwise vs Python path; NO_JIT identical |
| every row since 2026-09-26 | `seeds=stable-v1` | C-bench: yes; xinxin: our seats only | header token enforced by both harnesses |
