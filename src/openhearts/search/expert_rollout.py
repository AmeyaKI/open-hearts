"""Phase 7C: expert rulebooks as the OPPONENT playout policy inside honest search.

WHAT CHANGES, PLAINLY.  Honest search rehearses imagined futures; since Phase 1
every seat in that rehearsal has been played by the beginner heuristic.  Here
the three OPPONENT seats in the OUTER playouts are played by TRAIN-side
experts (players/expert.py, ids 1-200 -- never the sealed held-out 201-250):
the rehearsal opponent might win a cheap trick with the K♦ on purpose to
unload a liability, the way a real player would.  Nothing else moves:

  * our own seat keeps the beginner heuristic, plus the one-time inner
    re-determinization interception (n_inner > 0) exactly as before;
  * the inner SearchPlayer keeps heuristic playouts;
  * beliefs, the constraint sampler, and the world set are UNTOUCHED
    (`posterior_factory=None` is forced; the outer worlds are drawn by the
    inherited code BEFORE any expert identity is sampled);
  * `_use_fused()` is False (the fused kernel runs heuristic playouts in
    compiled code and cannot see this Python hook -- the silent-bypass trap
    the catalogue names); `group_equivalent` is False (grouping is proved
    for the old policy only).

IDENTITY SAMPLING (catalogue §5, plan 7C): one train identity per opponent
seat per OUTER WORLD, fixed through both playout segments and SHARED across
every candidate branch of that world (common random numbers: candidate A and
candidate B are judged against the same imagined people with the same tie
seeds).  One `rng.integers` draw per decision seeds all of it, taken inside
`_evaluate` AFTER the world sample so the world set is bitwise the
incumbent's.  With `rollout_ids=None` no draw is taken and the player IS the
incumbent (the reduction gate).

INFORMATION BOUNDARY.  A rollout expert receives `state.view_for(seat)` of
the SIMULATED state: its own sampled hand plus the simulated public history.
It never sees the other sampled hands or the root player's cards.

COST.  Pure Python: ~200 rule evaluations per playout.  Measured by the 7C
probe, never estimated; equal-time rows are pre-registered for exactly this
reason.
"""
import numpy as np

from openhearts.engine import cards
from openhearts.engine.state import GameState, PlayerView
from openhearts.players.expert import ExpertPlayer
from openhearts.players.expert_population import (
    DOMAIN_SEARCH, derive_seed, is_heldout, sample_expert,
)
from openhearts.search.decision import state_from_view
from openhearts.search.honest import HonestSearchPlayer


class ExpertRolloutSearchPlayer(HonestSearchPlayer):
    def __init__(self, level, n_outer, n_inner, rng, rollout_ids=None,
                 sampler_respects_voids=True, jit_sampler=True):
        super().__init__(level, n_outer, n_inner, rng,
                         sampler_respects_voids=sampler_respects_voids,
                         jit_sampler=jit_sampler, posterior_factory=None,
                         fused=False, group_equivalent=False)
        if rollout_ids is not None:
            rollout_ids = [int(i) for i in rollout_ids]
            assert rollout_ids, "empty rollout pool"
            leaked = [i for i in rollout_ids if is_heldout(i)]
            assert not leaked, f"held-out experts in the rollout pool: {leaked}"
            self._params = {i: sample_expert(i) for i in rollout_ids}
        self.rollout_ids = rollout_ids
        self._identities = None      # per outer world: {seat: (params, tie_seed)}
        self.rollout_playouts = 0
        self.last_identities = None

    def _use_fused(self) -> bool:
        return False

    def _evaluate(self, view: PlayerView, arrangements, candidates) -> int:
        """The inherited loop, plus per-world identities drawn once here."""
        if self.rollout_ids is None:
            return super()._evaluate(view, arrangements, candidates)
        base = int(self.rng.integers(2 ** 63))
        ids = self.rollout_ids
        idents = []
        for w in range(len(arrangements)):
            wr = np.random.default_rng(derive_seed(DOMAIN_SEARCH, base, w))
            picks = wr.integers(len(ids), size=3)
            seat_map = {}
            for i in range(3):
                seat = (view.seat + 1 + i) % 4
                seat_map[seat] = (self._params[ids[int(picks[i])]],
                                  derive_seed(DOMAIN_SEARCH, base, w, seat))
            idents.append(seat_map)
        self._identities = idents
        self.last_identities = idents

        base_score = view.scores[view.seat]
        avgs = []
        best_card, best_avg = None, None
        for card in candidates:
            total = 0
            for w, hands in enumerate(arrangements):
                state = state_from_view(view, hands)
                state.play(card)
                self._playout_expert(state, view.seat, idents[w])
                total += state.scores[view.seat] - base_score
            avg = total / len(arrangements)
            avgs.append(avg)
            if best_avg is None or avg < best_avg:
                best_card, best_avg = card, avg
        self.last_avgs = np.asarray(avgs, dtype=float)
        return best_card

    def _playout_expert(self, state: GameState, our_seat: int, seat_map) -> None:
        """Both outer segments with expert opponents; our seat heuristic with
        the one real-decision interception, exactly as `_playout_python`."""
        self.rollout_playouts += 1
        experts = {s: ExpertPlayer(np.random.default_rng(seed), params)
                   for s, (params, seed) in seat_map.items()}
        intercepted = self.n_inner <= 0
        while not state.is_over():
            seat = state.to_play
            view = state.view_for(seat)
            if seat == our_seat:
                if not intercepted and len(cards.cards_in(view.legal_moves)) > 1:
                    state.play(self._inner.choose(view))
                    intercepted = True
                    continue
                state.play(self._heuristic.choose(view))
            else:
                state.play(experts[seat].choose(view))
