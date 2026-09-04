# open-hearts

**A Hearts AI that beat the field's reference bot — by tracking exactly where every hidden card can be, then searching over realistic futures.**

Hearts is an imperfect-information card game: you never see the other players' hands. This engine keeps an exact, constraint-propagated probability table over every hidden card, updates it from hard evidence (a player who can't follow suit *provably* has none of that suit), samples concrete possible worlds from it, and picks the move that scores best across those futures.

Everything is measured like a research paper: matched deals, paired statistics, confidence intervals, expectations pre-registered *before* every run — and the experiments that failed are published next to the ones that won.

## Headline results

Hearts deals 26 penalty points per hand among 4 players — **6.5 points/hand is break-even, lower is better.**

| Opponent | Us | Them | Verdict |
|---|---|---|---|
| **xinxin** — the field's reference Hearts bot for 25 years (Sturtevant), at its published research configuration | **6.11** | 6.89 | **Won by 0.78 pts/hand** (95% CI [0.64, 0.91], Wilcoxon z = −11.3) over 14,000 games on the research literature's duplicate-seating protocol — **while deciding ~9× faster** (0.027 s vs 0.25 s per move, measured same-machine) |
| **OpenSpiel ISMCTS** (Google DeepMind's framework; generic untuned baseline, 1,000 simulations/move) | **4.70** | 7.10 | Won by 2.4 pts/hand over 1,000 matched deals — and the margin is **compute-robust**: raising the baseline's budget 10× (to ~50× our per-move cost) didn't close it |
| 3 held-out synthetic opponents (never trained against) | **2.35** | — | Takes ~9% of each hand's points while three opponents split the rest |
| **Exploitability** — a dedicated attacker built to beat our bot specifically | +0.20 pts/hand extra taken | — | The bot's own robustness, measured rather than assumed: a minor, quantified leak with a standing regression test |

All results are on the no-pass / no-shoot-the-moon variant, both sides playing identical rules (verified per-decision by runtime legality tripwires — zero firings across 18,000+ games).

## How it works, in one paragraph

When an opponent fails to follow suit, you learn something *exact*: they hold zero cards of that suit. Because hand sizes are fixed, that certainty ripples — probability mass moves between players and cards in a way that can be computed exactly (iterative proportional fitting over a 3-opponents × 52-cards table). The bot keeps that table exact at all times, samples complete hidden-hand arrangements from it, plays each imagined world forward, re-samples mid-future as new information would appear ("honest search" — so a move that *reveals* information gets credit for revealing it), and plays the card with the best average outcome.

## Engineering

- **20–30× end-to-end speedup** from 52-bit bitboard game representation and numba JIT-compiled kernels — measured, not estimated, across matched deals. A further fused decision kernel adds 5.3×.
- **Every optimization is provably safe:** each compiled kernel keeps its pure-Python reference implementation, tested and runnable (`OPENHEARTS_NO_JIT=1`), and deterministic paths are gated on **bitwise-identical** output before any speed number is quoted.
- **Architectural information hygiene:** player code can only receive a filtered `PlayerView` — it is structurally impossible for a bot to peek at hidden hands, not merely tested-for.
- **~17k lines** of Python across engine, inference, search, and evaluation; full test suite green in both JIT modes; a C++ backend port is in progress.
- Scale, on one laptop (M5 Max): 200,000 self-play games in ~2.5 minutes; a 14,000-game benchmark match in ~5 hours of checkpointed background compute.

## The method is the other half of the project

- **Pre-registration:** every experiment's expected outcome is written down and frozen before the run; results are reported against the original wording — including the misses.
- **Paired statistics:** every comparison plays identical deals with seats rotated, and confidence intervals bootstrap over deals. A built-in alarm (four identical players must average exactly 6.5) guards the whole harness.
- **Published negatives:** a learned position evaluator lost to the exact simulator five different ways; opponent-modeling ("reading") was measured to be worth approximately nothing against noisy opponents — and the initial "reading actively hurts" result was itself diagnosed down to a statistics bug (importance-weight collapse), corrected, and re-reported. The nulls shaped the design as much as the wins.
- **Adversarially reviewed claims:** benchmark wording is audited against what was actually measured — e.g. the xinxin win is stated for our rules variant, never inflated to full Hearts.

## What this is and isn't

**Is:** an evidence-first answer to "does exact belief tracking + honest search actually win?" — yes, against the strongest opponents obtainable: it beats the reference bot of the academic literature on that literature's own protocol, at a tenth of the decision cost.

**Isn't (yet):** full Hearts (no passing round, no moon-shooting — both sides play the same simplified rules), or tested against humans (a live-play relay is built and waiting). Both are on the roadmap, tracked with the same discipline.

**Full detail:** every phase, experiment, chart, null result, and all 22 numbered limitations live in **[docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)** — the complete research chronicle this page summarizes.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q                      # full suite
OPENHEARTS_NO_JIT=1 python -m pytest -q  # same suite on the pure-Python reference paths
```

Play against it or use it as an advisor at a real table:

```bash
python experiments/relay.py --advise   # bot suggests, you play
```

## Repository map

| Path | What lives there |
|---|---|
| `src/openhearts/engine/` | Bitboard game engine + numba kernels (pure-Python references retained) |
| `src/openhearts/belief/` | The exact belief table (constraint propagation, IPF rebalancing) |
| `src/openhearts/sampler/` | Constraint-respecting hidden-hand sampling |
| `src/openhearts/search/` | Honest (re-determinizing) Monte Carlo search — the champion bot |
| `experiments/` | Every benchmark and experiment, reproducible by seed |
| `docs/` | Charts + [the full research log](docs/RESEARCH_LOG.md) |
| `cpp/` | C++ backend port (in progress) |
