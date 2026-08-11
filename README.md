# open-hearts

A small project that plays the card game Hearts and tries to do two things honestly:

1. **Guess where the hidden cards are**, using only what a player would legitimately know (their own hand, and what's been played).
2. **Use those guesses to pick better moves**, by imagining a few likely arrangements of the hidden cards, playing each one out, and picking the move that scores best on average.

The whole point of this project is measuring whether step 2 actually works, and reporting the answer plainly even where it's underwhelming.

## The one-idea summary

In Hearts, once a player fails to follow suit (say, someone can't follow a club lead), you learn something exact: that player has **zero** clubs. That single fact doesn't just remove clubs from them — because everyone's hand size is fixed, all the probability that clubs *would* have taken up in their hand has to move somewhere else. We keep a running table of "how likely is each opponent to hold each card," update it every time a void is revealed, sample a few concrete hidden-hand guesses from that table, play each one out to the end with simple heuristic players, and choose the move that gives the lowest average points across those playouts.

## How the belief table works

The table is 3 opponents x 52 cards, holding P(opponent i has card c). Two constraints always hold:

- A card the observer can see (in their own hand, or already played) has probability 0 for every opponent.
- Each opponent's row must sum to the number of cards they're still holding.

**A worked example.** You are the observer, and the trigger is a player failing to follow suit. Concretely:

- West leads a heart. North discards a club instead of following. That's a **void reveal**: North has zero hearts.
- Before this: hearts were spread roughly evenly across North, East, and South's remaining hearts (since your own hand and played cards are zeroed out already).
- After this reveal: North's probability on every heart card drops to exactly 0. All of the probability mass that used to sit on "North holds this heart" has to go somewhere, because East and South must still collectively hold every remaining heart. So East's and South's probabilities on those hearts rise.
- But North still holds the same number of cards as before (nothing about her hand *size* changed) — so her probability on her *non-heart* cards has to rise too, to make her row sum back up to her hand size. She still has just as many cards, but now we know none of them are hearts, so the "missing" heart-probability she used to carry gets redistributed onto the clubs, diamonds, and spades she might hold.

This rebalancing — start from raw void/seen-card zeroing, then adjust rows and columns until both constraints (zeroed cells stay zero, each row sums to hand size) hold together — is implemented with iterative proportional fitting in `src/openhearts/belief/table.py`. It is exact for the constraints it's given: it correctly describes what opponents *could* be holding given voids and hand sizes. It says nothing about *how* opponents choose which card to play, which is a real limitation (see below).

There are three levels of belief, compared throughout this README:

- **UNIFORM** — ignore all evidence beyond "cards you can see are zero." Every unseen card equally likely for everyone.
- **VOIDS** — also use void reveals (the mechanism above), rebalanced.
- **FULL** — the full rebalanced table (voids plus proper row/column consistency).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run the fast test suite:

```bash
.venv/bin/python -m pytest -q
```

Run the slow fuzz tests (randomized, more thorough, take longer):

```bash
.venv/bin/python -m pytest -m slow -q
```

Run the experiments (results land in `results/`, which is gitignored — regenerate them rather than expecting them to be checked in):

```bash
python experiments/run_guessing.py          # Question A: are the guesses any good?
python experiments/run_ablation.py          # Question B: do better guesses win more points? (single core)
python experiments/run_ablation_parallel.py # same, but spread across multiple cores
python experiments/run_noleak.py            # corrected control, see "sampler void leak" below
python experiments/run_sampler_stats.py     # how hard the sampler has to work
```

## Headline result 1: can we guess where the cards are?

![Guessing accuracy by trick](docs/guessing.png)

We compare three belief levels against the true deal (which only the measurement code is allowed to see — the belief code never touches it). By trick 13 (the last trick, when there's the least uncertainty left):

| level   | mean probability on true holder | top-1 accuracy |
|---------|----------------------------------|-----------------|
| FULL    | 0.588                             | 0.60 |
| VOIDS   | 0.483                             | 0.48 |
| UNIFORM | 0.333                             | 0.33 |

At every trick from 1 to 13, FULL >= VOIDS >= UNIFORM. More evidence never hurts, and using void information clearly helps over doing nothing.

**Where we were wrong going in:** the plan guessed FULL would climb close to 1.0 by the last few tricks, since so few cards remain unseen. It doesn't — it plateaus around 0.59. We checked this against exact enumeration of all consistent hands, and 0.59 genuinely is the ceiling for this kind of evidence: in the last few tricks, the cards left in an opponent's hand are frequently *actually interchangeable* given only voids and hand-size constraints — there's no more information to extract from "what suits has this player failed to follow" alone. That ceiling is specific to this evidence class, though: real opponents aren't choosing cards uniformly at random from what's legal, they're playing according to some strategy, and inferring from *how* they play (not just what they're void in) could in principle push accuracy higher. We didn't build that — it's out of scope for this phase — so we can't say how much higher, only that the room for it exists.

## Headline result 2: do better guesses win more points?

![Ablation results](docs/ablation.png)
![Sample-count sweep](docs/sweep.png)

Setup: 500 deals, each played with the bot rotated through all 4 seats (2,000 games total), against 3 heuristic opponents. 26 points are dealt per hand, so 6.5 is the break-even point; scoring below 6.5 means beating heuristic opponents.

| config                     | mean  | 95% CI          |
|----------------------------|-------|-----------------|
| search-UNIFORM-n100        | 3.302 | [3.120, 3.485]  |
| search-VOIDS-n100          | 3.139 | [2.964, 3.314]  |
| search-FULL-n100           | 3.267 | [3.091, 3.447]  |
| heuristic (no search)      | 6.500 | (fixed)         |
| random-legal               | 11.548| [11.235, 11.859]|
| search-UNIFORM-noleak-n100 (corrected control) | 3.371 | [3.191, 3.554] |

**The honest reading:**

- Search helps, a lot. Any belief level with sampled-playout search (~3.2–3.4 points/hand) roughly halves the heuristic's score (6.5 points/hand), and the confidence intervals don't overlap. Just *searching* over sampled futures, even with a weak belief model, is clearly worth doing.
- The three belief levels **do not separate statistically** from each other. VOIDS is nominally lowest (3.139) and FULL is nominally worse than VOIDS (3.267), but their confidence intervals overlap heavily with UNIFORM's — we cannot say any belief level beats the others at n=100 samples.
- That's suspicious on its own, though, because even "UNIFORM" here was quietly getting some void information for free: the sampler (`src/openhearts/sampler/sampler.py`) refuses to deal a card into a suit an opponent is known to be void in, *at every belief level*, including UNIFORM. So "UNIFORM" wasn't really evidence-free. We built a corrected control (`search-UNIFORM-noleak-n100`, `sampler_respects_voids=False`) that truly ignores voids when sampling. It scored 3.371 — worse than the leaky UNIFORM (3.302) and worse than VOIDS/FULL, as expected — but its confidence interval still overlaps VOIDS and FULL, so even this doesn't establish a clean separation.
- **Conclusion, stated plainly:** with determinized search at 100 samples, belief sharpness beyond basic void-consistency is worth at most about 0.2 points per hand against opponents of this strength. That's a small, real effect that we can't confidently distinguish from noise at this sample size — not a negative result to hide, but exactly the kind of "it didn't move the needle much" finding the project asked to be reported rather than tuned away.

**Sample-count sweep** (FULL level, same 500 deals): 10 samples per decision is too few (mean 4.109 — noticeably worse); 50 samples captures most of the achievable value (3.314); going from 100 to 500 samples buys only about 0.12 more points (3.267 -> 3.143) for roughly 5x the compute. n=100 is the sweet spot used throughout.

## Sampler cost

We measured the plain restart-based sampler (`src/openhearts/sampler/sampler.py`: walk the unseen cards, assign each to an opponent by weighted draw, restart on a dead end) before considering any optimization, per the project's "measure before optimizing" rule. It never needed one: 0.0% failure rate at every trick, and even its worst case (trick 10, the most constrained point in the hand) needed on average only 1.15 attempts to find a valid arrangement. None of the sampler improvements sketched in planning were ever built, because the numbers said they weren't needed.

## Phase 2: making beliefs matter (honest search)

Phase 1 ended on a puzzle: the belief levels (UNIFORM/VOIDS/FULL) all scored about the same in points, even though FULL clearly knew more about the hidden cards. Something between "knowing more" and "playing better" was throwing that knowledge away.

**The flaw, in plain terms.** When the search imagines a hidden-card arrangement and plays it out to the end to see how many points a candidate move costs, that playout was using *one fixed, fully-known arrangement all the way to the end of the hand*. That's like planning your whole evening around a guess of exactly which bus your friend is on, instead of updating your plan once you actually see them get off a bus. A move whose value comes from *learning something* (e.g., leading a suit to see who's void in it) gets no credit for the learning, because the imagined opponents in that playout aren't uncertain about anything — they're reacting to a hand the search has already fully decided. So belief sharpness had nowhere to cash out: a smarter guess about the hidden cards can't help a plan that never re-checks its guess.

**The fix: honest search (one-level re-determinization).** `HonestSearchPlayer` (`src/openhearts/search/honest.py`) plays out one step into an imagined future, then — at the next decision point in that imagined future — throws away the rest of that one fixed arrangement and re-samples a *fresh* set of hidden-card guesses from what would actually be visible at that point (including anything newly revealed, like a fresh void). This is "one-level" because it only re-samples once per imagined path, not at every single card; a full fix would re-sample at every decision (this is "one rung below" full information-set search — see limitations).

**Results (ablation2.txt, same 500 deals as Phase 1, bot rotated through all 4 seats):**

| config                  | mean  | 95% CI          |
|-------------------------|-------|-----------------|
| honest-UNIFORM-noleak   | 3.469 | [3.270, 3.666]  |
| honest-VOIDS            | 3.343 | [3.151, 3.538]  |
| honest-FULL             | 3.253 | [3.070, 3.447]  |
| phase1-FULL-bridge      | 3.267 | [3.091, 3.447]  |

The honest reading:

- **The levels order correctly for the first time**: FULL (3.253) < VOIDS (3.343) < UNIFORM-uninformed (3.469). Under Phase 1's determinized search this ordering didn't hold at all.
- **The pre-registered criterion (marginal, non-overlapping CIs) was NOT met** — these three CIs still overlap each other. Judged only by that yardstick, this would be another null.
- **A sharper, legitimate test does show a real effect.** Because every configuration was run on identical deals (same 500 deals, same rotation), we can pair up FULL and UNIFORM's scores *deal by deal* rather than comparing them as two unrelated distributions — this is the same trick as a before/after study on the same patients instead of two different groups. That paired analysis shows FULL beats uninformed UNIFORM by **+0.215 points/hand, 95% CI (+0.03, +0.39)** — a confidence interval that excludes zero. This is the project's first statistically real belief-value result.
- **Honest search adds sensitivity, not raw strength.** Comparing honest-FULL (3.253) to Phase 1's determinized search-FULL (3.267) shows almost no difference: +0.013 points, not significant. Honest search didn't make the bot stronger overall — what it did was make the bot's score finally *respond* to how good its beliefs are (the belief-driven gap widened from a statistically invisible ~0.10 in Phase 1 to a real +0.215 here). Think of it as replacing a broken thermometer: the room isn't warmer, but now you can actually tell when it is.

**B-probe: predictability ruled out.** Before building honest search, we asked whether Phase 1's flatness was instead because the heuristic opponents are *too predictable* — if the search can already anticipate a deterministic opponent's exact response, sharper beliefs about their hand wouldn't add much. `results/bprobe.txt` reruns Phase 1's exact three search rows against `RandomizedHeuristic(epsilon=0.1)` opponents (10% chance of deviating from the heuristic on each decision) instead of the plain deterministic heuristic. The levels stayed just as flat (UNIFORM-noleak 3.164, VOIDS 3.155, FULL 2.986 — all overlapping), which rules out predictability as the explanation and pointed squarely at the resolved-future flaw that honest search above then fixed.

![Phase 2 ablation: honest search makes belief levels order correctly](docs/ablation2.png)
*Points per hand under honest search, by belief level. Lower is better. The order (FULL < VOIDS < UNIFORM) now matches what the belief table itself says about how much evidence each level uses — it didn't in Phase 1.*

### A JIT note

Honest search is expensive — playing out imagined games and then re-imagining more games inside them multiplies the work. Profiling showed the playout logic (not the belief math) was the bottleneck, so phases 2.5–2.7 and 3.5–3.6 replaced the hot inner loops (the heuristic playout, the sampler, the belief rebalance, and the choice-evidence audit) with `numba`-compiled kernels operating on the same bitmask representation, fused where the profiler showed it mattered. This bought roughly **34x** on search games end-to-end (an honest-search game went from 23.9s to 0.70s) with no change to what's being computed — every ported piece was checked against the original Python: bitwise-identical output wherever the computation is deterministic (the playout, the belief rebalance), and statistically indistinguishable within tolerance where randomness order can legitimately differ (the batch sampler, the choice-evidence audit). The original pure-Python code paths are kept and still tested; set `OPENHEARTS_NO_JIT=1` to run them instead of the compiled kernels, if you want to verify results without trusting the compiler.

## Phase 3: reading opponents' choices, not just their inabilities

**The idea.** The belief table (Phase 1/2) only uses *involuntary* evidence: what a player is void in, forced out by the game itself. But our heuristic opponents are deterministic — the same hand always produces the same play — which means every *voluntary* choice they make also leaks information. If the heuristic always discards its highest heart when it can't follow suit, then seeing it discard the 8♥ doesn't just show one card — it proves that opponent isn't holding anything higher (9♥ through A♥). That's evidence the constraint-only belief table (voids + hand sizes) structurally cannot see, because it never looks at *which* card among several legal ones got played.

**The mechanism: likelihood-weighted world filtering.** `WeightedPosterior` (`src/openhearts/belief/weighted.py`) samples many candidate hidden-hand arrangements the same way Phase 1's sampler did, but then asks of each one: "if this arrangement were true, and the opponents played their real policy, how likely is the sequence of moves we actually observed?" Arrangements that would have made the opponent play differently than they actually did get down-weighted (in "strict" mode, epsilon=0, down to exactly zero); the survivors are a set of worlds consistent not just with the rules, but with the *choices* actually made. An `epsilon` knob lets the audit tolerate some fraction of "off-policy" deviation, for when the opponent model is known to be imperfect.

**Headline: the 0.588 ceiling is broken.** Phase 1 found that constraint-only evidence provably tops out around 0.588 mean-probability-on-truth by trick 13 (matched to exact enumeration). Reading choices breaks that ceiling decisively (`guessing2.txt`, 500 games, all 4 seats):

| trick | CHOICE meanP | CHOICE top-1 |
|-------|--------------|--------------|
| 9     | 0.590        | 0.627        |
| 13    | 0.899        | 0.904        |

CHOICE crosses the old 0.588 ceiling by **trick 9** and reaches **0.899 mean probability on the true card / 90% top-1 accuracy** by trick 13 — a large, clean improvement over FULL's 0.588 / 0.60 at the same point.

**The partial null, stated plainly.** We pre-registered an expectation that CHOICE would approach ~1.0 by trick 13, since a fully deterministic opponent's choices should eventually pin down their hand almost exactly. It doesn't — it plateaus at 0.899, not 1.0. We checked the obvious suspect first: sampler starvation (running out of worlds to sample from). It isn't that — trick 13 still runs at a full 100 effective worlds, no thinning. The second thing we checked: of the arrangements that survive to the end, only about 41% turn out to be genuinely choice-consistent with everything observed, meaning even a hand-picking deterministic heuristic doesn't leak *everything* about its hand through its choices — some of its legal decisions are genuinely tied among cards that would have produced identical play, so no amount of choice-reading can separate them. This is a real, structural ceiling of this evidence class, not a bug or a starved sampler.

*A metric caveat:* the NLL column in `guessing2.png`'s four-line panel excludes "truth-zero" cards (cases where none of the sampled worlds happened to include the true holder for a given card — possible with a finite number of worlds, not a floor or a smoothing trick). That makes CHOICE's NLL line look better than a fully fair comparison would, since it's dropping exactly its hardest cases; meanP and top-1 are unaffected (a truth-zero counts as 0.0 / a miss, not excluded) and are the metrics to trust for the headline comparison above.

![Choice-aware guessing breaks the constraint-only ceiling](docs/guessing2.png)

![Top-1 vs top-2 accuracy: by the last trick the choice guesser's top two candidates contain the truth 99% of the time](docs/topk.png)

*(Top-3 is trivially 100% with only three opponents. The striking number here: CHOICE top-2 reaches 0.99 at trick 13 — the guesser's residual uncertainty is almost always between its two leading candidates, never a wild miss.)*
*Mean probability on the true card, top-1 accuracy, and NLL by trick, all four belief levels. CHOICE (reading opponents' choices) visibly separates from FULL (constraints only) starting mid-hand and crosses the old 0.588 ceiling around trick 9.*

**Survival (how often the world-filtering search runs dry).** Filtering out inconsistent worlds means some decisions can come up empty — no sampled world survives. `survival.txt` measures this under strict filtering (epsilon=0) across 300 games: the hardest stretch is tricks 6–9, where about **25% of decisions hit budget exhaustion at trick 8** (the single worst point) but total collapse — meaning *every* attempt fails and the code has to fall back — stays under 1% everywhere. The escalation mechanism sketched in planning (raising the sampling budget partway through) turned out to be unnecessary once the JIT speedup landed — the plain fixed budget was cheap enough to just run.

**Robustness: strict filtering is a double-edged sword against imperfect opponents.** Everything above assumes the opponents are exactly the deterministic heuristic the posterior is auditing against. `robustness.txt` tests what happens when that assumption is wrong: 300 games against opponents that deviate 10% of the time (`RandomizedHeuristic(epsilon=0.1)`), scored by three guessers — FULL (constraint-only, unaffected by the opponent model), CHOICE-strict (epsilon=0, assumes zero deviation), and CHOICE-soft (epsilon=0.1, matching the true deviation rate numerically, though it smooths differently than the true noise — see the file's caveat). The result: strict filtering **collapses in 96% of decisions by trick 13**, and once collapses are scored honestly as zero information (not skipped), strict is **worse than doing nothing but constraints from trick 3 onward**. Soft filtering (epsilon=0.1) never collapses and reaches 0.88 mean probability vs constraint-only's 0.60 at trick 13. The lesson: a choice-reading model that assumes a perfect opponent model is a trap against any opponent that isn't perfectly predictable — always carry some epsilon slack, even if you don't know the true deviation rate exactly.

![Strict choice-filtering degrades badly against opponents who deviate; soft filtering with slack does not](docs/robustness.png)
*FULL (constraint-only), CHOICE-strict (assumes a perfect deterministic opponent), and CHOICE-soft (epsilon=0.1 slack) against opponents that actually deviate 10% of the time. Strict collapses; soft holds up.*

**Full pipeline: choice-aware inference feeding honest search.** `ablation3.txt` puts it all together — `HonestSearchPlayer` whose outer worlds are drawn from `WeightedPosterior` (choice-aware, epsilon=0, which is the *correct* model here since these opponents really are the plain deterministic heuristic) instead of the raw constraint sampler, on the same 500 deals as the Phase 1/2 ablations:

| config          | mean  | 95% CI          |
|-----------------|-------|-----------------|
| honest-CHOICE   | 2.869 | [2.703, 3.038]  |
| honest-FULL     | 3.253 | [3.070, 3.447]  |

Paired per-deal (same identical-deals trick as Phase 2): honest-CHOICE beats honest-FULL by **+0.385 points/hand, 95% CI (+0.211, +0.561)** — a clear, non-null improvement from reading choices, stacked on top of Phase 2's honest search. `ablation3.txt` also reports that 44 of the run's decisions (0.2% of the ~20,700 total) hit a full posterior collapse and silently-forbidden-fallback rule kicked in, falling back to the plain constraint sampler for that one decision only — small, but stated rather than hidden, per the project's truth-safety rule.

## Current performance at a glance

![The search ladder: from random play (11.5 pts/hand) to the full pipeline (2.87)](docs/search_ladder.png)

![Full-pipeline ablation row: choice-aware beliefs through honest search](docs/ablation3.png)

One axis, whole story: random legal play eats 11.5 points a hand, a competent rule-follower 6.5, any sampled-playout search roughly halves that, and choice-aware beliefs through the honest search reach 2.87 — an 11% share of each hand's 26 points, against opponents taking ~30% each.

## Known limitations

Stated plainly, not hidden:

1. **Sampling bias.** The sampler assigns cards to opponents one at a time, weighted by the belief table's per-card probabilities. This is close to, but not exactly, sampling whole hidden-hand arrangements uniformly at random from everything consistent with the constraints. In practice it's close enough to be useful, but it is a real, acknowledged approximation.
2. **Playout opponents can effectively see everything (Phase 1 search; largely addressed by Phase 2's honest search, see above).** Inside one imagined arrangement, the playout code plays out a fully-specified game — which means the heuristic "opponents" in that imagined playout are reacting to a fully known hand, not the same partial information a real opponent would have. Honest search re-determinizes one level deep, which is why belief quality now shows up in points, but it is not a full fix (see limitation 7 below).
3. **The playout policy is a mirror.** Playouts use the exact same heuristic that the actual opponents use. That's a perfectly accurate opponent model by construction, which flatters the results — a real opponent might play very differently from the heuristic, and search tuned against a perfect mirror of its opponents won't necessarily transfer. Phase 3's choice-evidence machinery has this same dependency in a sharper form: see limitation 8.
4. **Passing and shooting the moon are out of scope.** The 3-card pass before play starts isn't modeled, and shooting the moon (taking all 26 points to flip the score) is ignored entirely — scoring always assumes the normal 26-points-split rule. The 6.5-point break-even control depends on both of these choices; a game with passing and moon-shooting could look different.
5. **Guessing curves are heuristic-specific.** The guessing accuracy numbers above (0.588 mean probability, 0.60 top-1 at trick 13 for FULL; 0.899 / 0.90 for CHOICE) were measured on games where every seat plays the same heuristic strategy. A different mix of opponents would produce different play patterns and could shift these numbers up or down.
6. **The rebalancing tolerance is a known approximation.** FULL-level rebalancing (iterative proportional fitting) accepts a 1e-4 numerical tolerance in cases where the constraints force some entries to the boundary (exactly 0). We verified this stays within about 0.003 of exact enumeration on the cases we checked, which is small, but the principled fix — detecting forced placements directly instead of approaching them iteratively — wasn't built this phase.
7. **Honest search is one rung below full information-set search.** One-level re-determinization re-samples once per imagined path, at the next decision point. A full fix would re-sample at every single decision inside every imagined playout (closer to true information-set MCTS), which would likely earn more of the sensitivity honest search unlocked but was out of scope for this phase on cost grounds.
8. **Choice evidence assumes a known opponent policy.** `WeightedPosterior` in strict mode (epsilon=0) is only exactly correct when the real opponents play exactly the modeled deterministic heuristic. Against opponents who deviate even slightly, strict filtering doesn't just weaken gracefully — the robustness experiment above shows it collapses hard (96% of decisions by trick 13) and can end up worse than not reading choices at all. The epsilon knob quantifies the cost of a wrong model and gives a way to hedge against it, but epsilon itself has to be guessed at or measured in advance — there's no mechanism here for learning it online.
9. **Soft-mode weight degeneracy.** CHOICE-soft (epsilon>0) computes each world's weight as a product of many small per-decision factors, so even when 100 worlds nominally survive, the *effective* number carrying most of the weight can be as low as roughly 1–6 (`mean_n_effective` in robustness.txt) — a handful of arrangements can end up dominating the posterior even though many are technically alive. Soft mode is still far more robust than strict, but its stated 100-world budget partly overstates how much genuine diversity is being sampled from.
