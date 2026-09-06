"""Exact expert-policy likelihood and the two 7B ORACLE world sources.

WHAT THIS IS (PHASE7_PLAN.md 7B pre-registration §D3, the catalogue §5
interface).  Phase 7B hands the champion the held-out experts' TRUE decision
rules and asks whether reading such opponents helps once the estimator is not
starved.  The likelihood here is EXACT: at every opponent decision in a
candidate world we replay that opponent's rulebook on the view it would have
had in that world and take P(observed card).  For a `tie_mode='fixed'`
expert that is 1 or 0; for `seeded_uniform` it is 1/|tie set| or 0.  A LEGAL
card can therefore get probability zero -- the old softmax profiler's strict
positivity does not carry over, and that is the point: this is not a fitted
profiler and is never labelled as one.

INFORMATION BOUNDARY.  Replay reconstructs each hypothetical actor's OWN
hand and public view from the candidate world (`_reconstruct_original_hands`)
-- never the real hidden hands.  The observer's own plays contribute nothing
(we know why we played them).

TWO WORLD SOURCES, same likelihood:
  * `expert_oracle_factory` -- the plain archived-ORACLE shape: draw N
    candidates from the constraint sampler, keep the survivors, stop at N
    survivors or `max_draws`.  With mostly 0/1 weights, degeneracy shows as
    WORLD STARVATION (few survivors), and zero survivors raises
    `PosteriorCollapse`, which honest.py counts and falls back on -- never
    silently.
  * `expert_sir_factory` -- A1's remedy: `search.sir.sir_posterior` with this
    likelihood plugged into its `logweight_fn` seam; the pool grows until the
    ESS target (≈ survivor count here) or the cap, then N equal-weight worlds
    are resampled.  Cap hits, distinct pools and survivors per trick are
    recorded exactly as A1 recorded them.

Both are Python reference paths in both JIT modes; no kernel exists and none
is claimed.  Optimizations that keep the arithmetic identical: forced plies
(one legal card) are never scored; the replay stops at the first zero factor;
the actor's view is built once per ply.
"""
import math

import numpy as np

from openhearts.belief.table import BeliefTable
from openhearts.belief.weighted import (
    PosteriorCollapse, WeightedPosterior, _assert_two_clubs_leads,
    _reconstruct_original_hands,
)
from openhearts.engine import cards
from openhearts.engine.game import legal_moves
from openhearts.engine.state import GameState, PlayerView
from openhearts.players.expert import action_distribution
from openhearts.search import sir as _sir

NEG_INF = float("-inf")


class ExpertLikelihood:
    """P(observed opponent plays | world) under the experts' TRUE records.

    `seat_params`: {seat: ExpertParams} for the three opponent seats; set (or
    reset) per game by the harness hook, exactly as the archived ORACLE row
    sets its per-seat parameter vectors.  Records are used in-process only;
    nothing here prints or serialises them (the held-out wall).
    """

    def __init__(self, seat_params=None):
        self.seat_params = dict(seat_params or {})
        self.n_scored = 0      # opponent decisions scored (diagnostic)
        self.n_zero = 0        # worlds killed by a zero factor
        self.n_illegal = 0     # worlds killed by an illegal observed play

    def world_logweight(self, view, world_hands) -> float:
        hands, all_plays = _reconstruct_original_hands(view, world_hands)
        if not all_plays:
            return 0.0
        _assert_two_clubs_leads(hands, all_plays)
        state = GameState(hands=hands)
        state.to_play = all_plays[0][0]
        observer = view.seat
        total = 0.0
        for seat, card in all_plays:
            assert seat == state.to_play, (
                f"replay desync: observed seat {seat} != {state.to_play}")
            legal = legal_moves(state.hands[seat], tuple(state.current_trick),
                                state.hearts_broken, state.trick_number)
            if not (legal & cards.bit(card)):
                self.n_illegal += 1
                return NEG_INF
            if seat != observer and (legal & (legal - 1)):   # 2+ legal cards
                params = self.seat_params[seat]
                actor_view = PlayerView(
                    seat=seat, hand=state.hands[seat],
                    history=tuple(state.history),
                    current_trick=tuple(state.current_trick),
                    hearts_broken=state.hearts_broken,
                    trick_number=state.trick_number,
                    scores=tuple(state.scores), legal_moves=legal)
                cs, ps = action_distribution(actor_view, params)
                self.n_scored += 1
                p = 0.0
                for c, x in zip(cs, ps):
                    if c == card:
                        p = x
                        break
                if p <= 0.0:
                    self.n_zero += 1
                    return NEG_INF
                total += math.log(p)
            state.play(card)
        return total


def expert_logweight_fn(view, world_hands, lik) -> float:
    """The `sir_posterior(logweight_fn=...)` seam shape."""
    return lik.world_logweight(view, world_hands)


def expert_oracle_posterior(view, level, lik: ExpertLikelihood, n_worlds: int,
                            rng, max_draws: int, keep_worlds: bool = True,
                            recorder=None) -> WeightedPosterior:
    """Plain ORACLE: constraint-sampler candidates filtered by the exact
    likelihood, first `n_worlds` survivors (in draw order) or `max_draws`."""
    table = BeliefTable.from_view(view, level)
    if not cards.cards_in(table.unseen_mask):
        return WeightedPosterior(np.zeros((3, 52)), list(table.opponent_seats),
                                 table.unseen_mask, list(table.hand_sizes),
                                 0.0, 0, 0, 0.0)
    batched = _sir.kernel.jit_enabled()
    worlds, logs = [], []
    draws = 0
    while len(worlds) < n_worlds and draws < max_draws:
        want = min(max_draws - draws, n_worlds - len(worlds))
        candidates, used = _sir._draw_candidates(table, rng, want, batched)
        draws += used
        for wh in candidates:
            lw = lik.world_logweight(view, wh)
            if lw == NEG_INF:
                continue
            worlds.append([int(wh[i]) for i in range(3)])
            logs.append(float(lw))
            if len(worlds) >= n_worlds:
                break
    if not worlds:
        raise PosteriorCollapse(
            f"no candidate world survived the exact expert likelihood after "
            f"{draws} draws (level={level})")
    lg = np.asarray(logs, dtype=np.float64)
    w = np.exp(lg - lg.max())
    ess = _sir.kish_ess(w)
    if recorder is not None:
        recorder.observe(
            trick=view.trick_number, m=len(worlds), ess_plain=ess,
            ess_pool=ess, distinct_pool=len({tuple(x) for x in worlds}),
            distinct_resampled=len({tuple(x) for x in worlds}),
            cap_hit=len(worlds) < n_worlds, n_illegal=0,
            n_underflow=int((w == 0.0).sum()), draws=draws, stages=1)
    weights = [float(x) for x in w]
    return WeightedPosterior(
        _sir._probs_from_worlds(worlds, w), list(table.opponent_seats),
        table.unseen_mask, list(table.hand_sizes), float(ess), draws,
        len(worlds), float(w.sum()),
        worlds if keep_worlds else None, weights if keep_worlds else None)


def expert_oracle_factory(lik: ExpertLikelihood, level, n_worlds=50,
                          max_draws=2000, recorder=None):
    """-> `f(view, rng) -> WeightedPosterior`, the shape honest.py wants."""
    def make(view, rng):
        return expert_oracle_posterior(view, level, lik, n_worlds, rng,
                                       max_draws, keep_worlds=True,
                                       recorder=recorder)
    make.likelihood = lik
    make.recorder = recorder
    return make


def expert_sir_factory(lik: ExpertLikelihood, level, n_worlds=50,
                       m_start=None, m_cap=2000, ess_target=None,
                       max_draws=None, resampler="systematic", recorder=None):
    """A1's SIR construction with the exact expert likelihood in the seam.
    `sir_posterior` itself is untouched; only its documented `logweight_fn`
    hook is used, so the archived rows cannot be affected."""
    max_draws = int(max_draws if max_draws is not None else 4 * m_cap)

    def make(view, rng):
        return _sir.sir_posterior(
            view, level, lik, n_worlds, rng, max_draws,
            ess_target=ess_target, m_start=m_start, m_cap=m_cap,
            resampler=resampler, keep_worlds=True, recorder=recorder,
            logweight_fn=expert_logweight_fn)
    make.likelihood = lik
    make.recorder = recorder
    return make
