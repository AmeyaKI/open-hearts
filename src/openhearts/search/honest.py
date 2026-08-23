"""Honest search: one-level re-determinization.

Phase 1's `SearchPlayer` (search/decision.py) evaluates every candidate inside
a fully-resolved sampled world: once a world is drawn, the imagined future
plays out as if we already knew every card. That is the *resolved-future
flaw*. Its consequence is not that the bot cheats -- the sampled world is
drawn only from the view -- but that the imagined *us* of the future is
assumed omniscient. A play whose whole value is that it teaches us something
(leading a suit to see who discards, holding a guard until the queen's
location is known) earns no credit, because in the imagined future we already
knew the answer. Symmetrically, hedging against ignorance looks pointless.
That is why Phase 1's belief levels did not separate in points: the evaluation
was indifferent to how sharp the belief was.

This player fixes that one rung up. After playing candidate X into an imagined
world, the playout runs normally until OUR seat's first subsequent real
decision (more than one legal move). At that point we do NOT let the scripted
world decide. We take the imagined state's *view* for our seat -- which now
contains the opponents' responses to X and nothing hidden -- build a fresh
belief table from it and re-determinize: sample `n_inner` new worlds and pick
the move that is best across them. In other words the value of X now includes
"what I will actually know after X, and what I will do with it". Probing plays
finally earn credit, and belief sharpness is exercised at both sampling
stages.

This is one rung below full information-set search (IS-MCTS), deliberately:
only the first future decision is re-determinized, and only for our own seat.
Every later decision, ours and the opponents', is plain heuristic, and the
opponents are still modelled by the exact policy they actually use. Those
remain known limitations, not fixed here.

Phase 3 (Task 9) adds an optional `posterior_factory`. When supplied, the
OUTER worlds no longer come from the raw constraint sampler but from a
`WeightedPosterior` -- worlds that survive being replayed against what the
opponents ACTUALLY chose. That closes the pipeline: choice-aware inference ->
honest search -> points. The INNER re-determinization deliberately stays
constraint-based, and that is not a budget compromise: inside an imagined
world the "observed" plays after our candidate are ones the playout itself
invented, so auditing them would only measure how well the imagined world
explains its own imagined plays. There is no choice evidence in there to read.

Phase 2.8 adds an optional `fused=True`. It changes no semantics and no
numbers: the whole candidate x world evaluation loop (including the inner
re-determinization) runs inside one compiled call instead of crossing the
Python boundary per playout segment.

It is BITWISE identical to the unfused JIT path, deliberately. The obvious
implementation -- seed numba's generator once per decision and draw a
continuous stream -- would have changed the rng stream and made verification
statistical only. Instead the orchestrator PRE-DRAWS the seeds: the outer
sample happens in Python exactly where it always did, and then the seeds for
the inner re-determinizations are drawn up front with the same scalar
`rng.integers(2**63)` call, consumed by the kernel in the same order, and the
Generator is rewound and advanced by exactly the number consumed. Same worlds,
same playouts, same per-candidate means, same card, same rng end state.
See `kernel.honest_decision`.

It is still DEFAULT OFF until the gates are signed off: every committed row --
the 2.869 bridge included -- stays reachable with the flag off, and
`OPENHEARTS_NO_JIT=1` never takes the fused branch at all.

An optional `group_equivalent=True` (ScrofaZero adoption (i), DEFAULT OFF)
cuts the candidate list to one representative per provably interchangeable
class -- same suit, nothing live between them in rank, no queen of spades in
the chain -- and credits the representative's score to the whole class. The
condition, the isomorphism proof, and the one caveat (which member we play is
readable at a real table, and grouping deletes the accidental randomness the
C0 probe measured on exactly these indifferent choices) live in
`search/grouping.py`. It is a theorem, not an approximation, but it is NOT
bitwise-compatible with existing rows: fewer candidates means fewer pre-drawn
inner seeds and a different rng end state, so it must never be mixed inside
one experiment.

Scope note for a future task: `ValueSearchPlayer` / `ProfiledSearchPlayer`
are NOT fused here. Their horizon path would hook in at the same place this
one does -- `_use_fused()` below -- by passing their fused scorer down into
the kernel's inner evaluation in place of `_inner_best`'s heuristic playouts.
That is deliberately left undone.

`n_inner=0` disables re-determinization entirely: the player then reduces
exactly to Phase-1 `SearchPlayer`, drawing the identical rng stream, so old
rows stay reproducible and the two can be compared on the same deals -- with
`jit_sampler=False`, since the compiled batch sampler (Phase 2.6, the default
here) deliberately draws a different, still deterministic, stream.
"""
import numpy as np

from openhearts.belief.table import BeliefTable, Level
from openhearts.belief.weighted import PosteriorCollapse
from openhearts.engine import cards, kernel
from openhearts.engine.state import GameState, PlayerView
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.sampler.sampler import sample_arrangement
from openhearts.search import grouping
from openhearts.search.decision import SearchPlayer, state_from_view


_LEVEL_CODE = {
    Level.UNIFORM: kernel.LEVEL_UNIFORM,
    Level.VOIDS: kernel.LEVEL_VOIDS,
    Level.FULL: kernel.LEVEL_FULL,
}


class HonestSearchPlayer:
    def __init__(self, level: Level, n_outer: int, n_inner: int, rng,
                 sampler_respects_voids: bool = True,
                 jit_sampler: bool = True,
                 posterior_factory=None,
                 fused: bool = False,
                 group_equivalent: bool = False):
        self.level = level
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.rng = rng
        self.sampler_respects_voids = sampler_respects_voids
        # Unlike SearchPlayer, honest search defaults to the compiled batch
        # sampler: it has no bitwise-pinned published rows to protect, and the
        # sampler is its dominant cost. OPENHEARTS_NO_JIT=1 still forces the
        # Python path.
        self.jit_sampler = jit_sampler
        # `posterior_factory(view, rng) -> posterior`, where `posterior` has
        # `.worlds` (a list of 3-element lists of CURRENT-hand bitmasks, in
        # BeliefTable.opponent_seats order -- exactly what the sampler
        # returns) and `.weights` (one float per world). None keeps the
        # Phase-2 behaviour bitwise: no extra rng is drawn, no extra branch
        # is taken.
        self.posterior_factory = posterior_factory
        # Phase 2.8, DEFAULT OFF until the lead flips it. Only ever consulted
        # through `_use_fused()` below, so NO_JIT mode, the Phase-1 reduction
        # (n_inner=0) and the Python-sampler config all keep their exact old
        # code path with the flag in either position.
        self.fused = fused
        # Equivalence-class card grouping (ScrofaZero adoption (i)),
        # DEFAULT OFF. When on, provably interchangeable legal cards are
        # evaluated once at the lowest member of their class and the score is
        # credited to the whole class. The theorem is in search/grouping.py.
        # It is NOT bitwise-compatible with existing rows: fewer candidates
        # means fewer pre-drawn inner seeds on the fused path and a different
        # rng end state (same card, by theorem, when the scores are matched).
        self.group_equivalent = group_equivalent
        # The candidate list the most recent decision actually evaluated, and
        # the class map behind it (diagnostics, for the gate tests).
        self.last_candidates = None
        self.last_rep_of = None
        # Per-candidate mean scores from the most recent decision
        # (diagnostic, for the gate tests). Set by the fused path, by the
        # unfused reference loop in `_evaluate`, and by the exploiter's
        # batched kernel -- so the three can be compared directly.
        self.last_avgs = None
        self.fallbacks = 0         # outer decisions that fell back to heuristic
        self.failed_samples = 0     # outer arrangements the sampler could not build
        # Decisions where the posterior found no surviving world and we fell
        # back to the constraint sampler. The plan's truth-safety rule calls
        # all-worlds-dead "a loud error, never a silent fallback": the
        # fallback here is deliberate but NOT silent -- it is counted, and
        # every experiment using it reports the count.
        self.posterior_collapses = 0
        # Decisions served by the posterior, and total worlds it supplied
        # (so a run can report the realised worlds/decision, which is <=
        # n_outer whenever the posterior ran out of draws).
        self.posterior_decisions = 0
        self.posterior_worlds = 0
        self._heuristic = HeuristicPlayer()
        # The inner evaluation is Phase-1 search, reused verbatim. It shares
        # our rng, so it must not exist at all when disabled: with n_inner=0
        # the rng stream has to match Phase 1's exactly.
        self._inner = None
        if n_inner > 0:
            self._inner = SearchPlayer(level, n_inner, rng,
                                       sampler_respects_voids, jit_sampler)

    def _use_fused(self) -> bool:
        """May this decision run the fused kernel? One place, overridable.

        All four conditions are load-bearing:
        - `fused`: opt-in, default off.
        - `n_inner > 0`: with no inner re-determinization there is nothing to
          fuse and the Phase-1 rng reduction must stay exact.
        - `jit_sampler`: the unfused inner would otherwise use the PYTHON
          sampler, which consumes the caller's Generator once per card. The
          kernel reproduces the compiled batch sampler's stream, not that one,
          so fusing there would silently change results.
        - `jit_enabled()`: `OPENHEARTS_NO_JIT=1` forces the reference paths.

        Subclasses that hook into `_playout` (the exploiter's nested champion)
        MUST override this to return False -- the fused kernel runs its own
        playouts in compiled code and cannot see a Python hook.
        """
        return (self.fused and self.n_inner > 0 and self.jit_sampler
                and kernel.jit_enabled())

    @property
    def inner_fallbacks(self) -> int:
        return self._inner.fallbacks if self._inner is not None else 0

    @property
    def inner_failed_samples(self) -> int:
        return self._inner.failed_samples if self._inner is not None else 0

    def choose(self, view: PlayerView) -> int:
        legal = cards.cards_in(view.legal_moves)
        if len(legal) == 1:
            return legal[0]

        arrangements = None
        if self.posterior_factory is not None:
            arrangements = self._posterior_worlds(view)

        if arrangements is None:
            table = BeliefTable.from_view(view, self.level)
            if not self.sampler_respects_voids:
                table = BeliefTable(table.probs, [set(), set(), set()],
                                    table.hand_sizes, table.opponent_seats,
                                    table.unseen_mask)
            arrangements = self._sample(table, self.n_outer)

            # This guard belongs to the SAMPLER only. A short posterior
            # return is not a degenerate draw: 20 choice-consistent worlds
            # are better evidence than 50 merely constraint-consistent ones,
            # and far better than punting to the heuristic.
            if len(arrangements) * 2 < self.n_outer:
                self.fallbacks += 1
                return self._heuristic.choose(view)

        candidates = legal
        self.last_rep_of = None
        if self.group_equivalent:
            rep_mask, rep_of = grouping.equivalence_classes(
                view.hand, view.legal_moves,
                grouping.dead_mask_from_view(view))
            candidates = cards.cards_in(rep_mask)
            self.last_rep_of = rep_of
        self.last_candidates = candidates

        if self._use_fused():
            # Steps 3-4 in one crossing. Same worlds in (constraint sampler or
            # posterior alike), same seeds, same tie-break out -- bitwise.
            card, n_fb, n_fs, avgs = kernel.honest_decision(
                view, arrangements, view.seat, self.n_inner,
                _LEVEL_CODE[self.level], self.sampler_respects_voids,
                self.rng,
                candidates=None if not self.group_equivalent else candidates)
            # Keep `inner_fallbacks` / `inner_failed_samples` reporting the
            # same quantities they report on the unfused path.
            self._inner.fallbacks += n_fb
            self._inner.failed_samples += n_fs
            self.last_avgs = avgs
            return card

        return self._evaluate(view, arrangements, candidates)

    def _evaluate(self, view: PlayerView, arrangements, candidates) -> int:
        """Steps 3-4, the reference (unfused) loop: score every candidate.

        Split out of `choose` in Phase 6C as the ONE overridable hook for an
        alternative evaluator -- the exploiter's batched nested-model kernel
        overrides exactly this method. Everything before it (the belief table,
        the outer world draw, the fallback guard, the candidate list) stays
        inherited code that no subclass touches, which is what keeps the
        exploiter's gate (iii) -- "the model never moves beliefs or the
        sampler" -- a structural fact rather than a promise.

        The loop itself is unchanged apart from recording `last_avgs`, a bare
        diagnostic assignment that draws nothing and branches on nothing, so
        the fused path's per-candidate means now have an unfused counterpart
        to be compared against.
        """
        base_score = view.scores[view.seat]
        avgs = []
        best_card, best_avg = None, None
        for card in candidates:
            total = 0
            for hands in arrangements:
                state = state_from_view(view, hands)
                state.play(card)
                self._playout(state, view.seat)
                total += state.scores[view.seat] - base_score
            avg = total / len(arrangements)
            avgs.append(avg)
            if best_avg is None or avg < best_avg:
                best_card, best_avg = card, avg
        self.last_avgs = np.asarray(avgs, dtype=float)
        return best_card

    def _posterior_worlds(self, view: PlayerView):
        """Outer worlds from the choice-aware posterior, or None to fall back.

        Weights: at epsilon = 0 every surviving world has weight exactly 1.0
        (the likelihood factors are 0/1, so survival means all-ones), and the
        kept worlds are an unbiased sample to be used as-is -- which is the
        configuration Task 9 runs, because the real opponents ARE the exact
        deterministic policy being audited. If a caller supplies epsilon > 0
        the weights are no longer uniform, so using the worlds directly would
        silently misweight the average; we resample n_outer of them with
        replacement proportional to weight, using this player's own rng
        (deterministic given its seed). That is a correct-in-expectation but
        higher-variance path, and it is not what any committed experiment
        runs.
        """
        try:
            posterior = self.posterior_factory(view, self.rng)
        except PosteriorCollapse:
            # Counted, never silent -- see the counter's comment.
            self.posterior_collapses += 1
            return None

        worlds = [[int(h) for h in w] for w in posterior.worlds]
        weights = [float(w) for w in posterior.weights]
        assert len(weights) == len(worlds), "posterior weights/worlds mismatch"
        if not worlds:
            # NOT a collapse: a collapse raises. An empty list here means the
            # posterior survived but retained nothing -- i.e. the factory
            # forgot `keep_worlds=True`. Counting that as a collapse would let
            # a row run entirely on the constraint sampler while still calling
            # itself choice-aware, so it is a loud error instead.
            raise AssertionError(
                "posterior returned no worlds despite surviving "
                f"({getattr(posterior, 'n_worlds_used', '?')} worlds kept); "
                "the factory must pass keep_worlds=True"
            )

        self.posterior_decisions += 1
        # Counted BEFORE any resampling, so it reports what the posterior
        # actually supplied rather than the resampled n_outer.
        self.posterior_worlds += len(worlds)

        if any(w != weights[0] for w in weights):
            probs = np.asarray(weights, dtype=float)
            probs /= probs.sum()
            idx = self.rng.choice(len(worlds), size=self.n_outer, p=probs)
            worlds = [worlds[i] for i in idx]
        return worlds

    def _sample(self, table, n_samples: int):
        """Same contract and counter semantics as SearchPlayer._sample."""
        if self.jit_sampler and kernel.jit_enabled():
            arrangements, n_failed = kernel.sample_arrangements(
                table, self.rng, n_samples)
            self.failed_samples += n_failed
            return arrangements
        arrangements = []
        for _ in range(n_samples):
            result = sample_arrangement(table, self.rng)
            if result is None:
                self.failed_samples += 1
                continue
            hands, _attempts = result
            arrangements.append(hands)
        return arrangements

    def _playout(self, state: GameState, our_seat: int) -> None:
        if kernel.jit_enabled():
            self._playout_jit(state, our_seat)
            return
        self._playout_python(state, our_seat)

    def _playout_jit(self, state: GameState, our_seat: int) -> None:
        # Same three segments as the Python loop, with the two heuristic-only
        # stretches handed to the compiled kernel: run to our first real
        # decision, re-determinize there in Python, then finish.
        if self.n_inner > 0:
            if kernel.run_playout_until_decision(state, our_seat):
                state.play(self._inner.choose(state.view_for(our_seat)))
        kernel.run_playout(state)

    def _playout_python(self, state: GameState, our_seat: int) -> None:
        # Reference implementation, kept live for OPENHEARTS_NO_JIT=1.
        # `intercepted` is deliberately a local: the one re-determinization is
        # per (world x candidate) playout, not per decision or per player.
        intercepted = self.n_inner <= 0
        while not state.is_over():
            seat = state.to_play
            view = state.view_for(seat)
            if not intercepted and seat == our_seat:
                # Only a real decision consumes the interception; at a forced
                # move there is nothing to re-determinize.
                if len(cards.cards_in(view.legal_moves)) > 1:
                    state.play(self._inner.choose(view))
                    intercepted = True
                    continue
            state.play(self._heuristic.choose(view))
