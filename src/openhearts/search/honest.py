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

`n_inner=0` disables re-determinization entirely: the player then reduces
exactly to Phase-1 `SearchPlayer`, drawing the identical rng stream, so old
rows stay reproducible and the two can be compared on the same deals -- with
`jit_sampler=False`, since the compiled batch sampler (Phase 2.6, the default
here) deliberately draws a different, still deterministic, stream.
"""
from openhearts.belief.table import BeliefTable, Level
from openhearts.engine import cards, kernel
from openhearts.engine.state import GameState, PlayerView
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.sampler.sampler import sample_arrangement
from openhearts.search.decision import SearchPlayer, state_from_view


class HonestSearchPlayer:
    def __init__(self, level: Level, n_outer: int, n_inner: int, rng,
                 sampler_respects_voids: bool = True,
                 jit_sampler: bool = True):
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
        self.fallbacks = 0          # outer decisions that fell back to heuristic
        self.failed_samples = 0     # outer arrangements the sampler could not build
        self._heuristic = HeuristicPlayer()
        # The inner evaluation is Phase-1 search, reused verbatim. It shares
        # our rng, so it must not exist at all when disabled: with n_inner=0
        # the rng stream has to match Phase 1's exactly.
        self._inner = None
        if n_inner > 0:
            self._inner = SearchPlayer(level, n_inner, rng,
                                       sampler_respects_voids, jit_sampler)

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

        table = BeliefTable.from_view(view, self.level)
        if not self.sampler_respects_voids:
            table = BeliefTable(table.probs, [set(), set(), set()],
                                table.hand_sizes, table.opponent_seats,
                                table.unseen_mask)
        arrangements = self._sample(table, self.n_outer)

        if len(arrangements) * 2 < self.n_outer:
            self.fallbacks += 1
            return self._heuristic.choose(view)

        base_score = view.scores[view.seat]
        best_card, best_avg = None, None
        for card in legal:
            total = 0
            for hands in arrangements:
                state = state_from_view(view, hands)
                state.play(card)
                self._playout(state, view.seat)
                total += state.scores[view.seat] - base_score
            avg = total / len(arrangements)
            if best_avg is None or avg < best_avg:
                best_card, best_avg = card, avg
        return best_card

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
