"""B1 (ROADMAP post-Phase-7 program): domain-aware evaluators for OpenSpiel's ISMCTS.

OpenSpiel's `ISMCTSBot` takes an `Evaluator` with two hooks -- `evaluate(state)`
(value of a determinized full-information state, one number per seat, in the
game's return units) and `prior(state)` (a distribution over legal actions used
by PUCT and by expansion). The stock `RandomRolloutEvaluator` plays random cards
to the end. This module plugs OUR Phase-1 beginner heuristic into both hooks so
the tree search is the same third-party algorithm with Hearts knowledge in it:

  HeuristicRolloutEvaluator(prior_eps=None)
      evaluate: rebuild our GameState from the OpenSpiel history (the deal is
                history()[1:53], holder = (index-1) % 4; plays follow), finish
                the deal with `kernel.run_playout` -- the bitwise-gated JIT
                heuristic playout, all four seats -- and return 26 - points.
      prior:    uniform over legal actions when prior_eps is None (rung 2);
                otherwise (1 - eps) on the heuristic's card and eps/|legal| on
                every legal card (rung 3, PUCT).

RETURN UNITS. OpenSpiel's `state.returns()` is 26 - points except when one seat
takes all 26, where it applies the shoot-the-moon rule (shooter 26, others 0).
This evaluator uses OUR no-moon scoring throughout (26 - points, always), which
is the variant every reported number is scored in; the stock rung's tree thus
values a moon shot the reported scoring never pays -- a disclosed baseline flaw
that rungs 2-3 do not inherit.

Deterministic: no rng anywhere; the ISMCTS bot's own random_state supplies all
sampling. Never touches the ISMCTS bot's hidden information: the state handed
to both hooks is the determinized world the tree is already searching.
"""
import numpy as np

from open_spiel.python.algorithms.mcts import Evaluator

from openhearts.engine import cards, kernel
from openhearts.engine.state import GameState
from openhearts.players.heuristic import HeuristicPlayer

from . import adapter

DEAL_START = 1          # history()[0] is the (inert) pass-direction chance action
DEAL_END = 53           # history()[1:53] are the 52 dealt cards


def _play_unchecked(m: GameState, card: int) -> bool:
    """GameState.play without the legality assertion; returns whether the play
    WAS legal under our rules. OpenSpiel's `resample_from_infostate` redeals the
    unseen cards subject to hand sizes only, so the recorded public plays can be
    illegal in the resampled world (a heart discarded on trick 1 by a seat that
    now holds clubs; a non-follow by a seat that now holds the led suit). The
    stock ISMCTS searches those worlds too; we evaluate the world we are given
    and COUNT the inconsistency (report-only diagnostic, PHASE7_PLAN.md B1)."""
    from openhearts.engine.game import legal_moves, trick_winner, trick_points
    seat = m.to_play
    legal = legal_moves(m.hands[seat], tuple(m.current_trick), m.hearts_broken, m.trick_number)
    was_legal = bool(legal & cards.bit(card))
    m.hands[seat] &= ~cards.bit(card)
    m.current_trick.append((seat, card))
    if cards.suit(card) == cards.HEARTS:
        m.hearts_broken = True
    if len(m.current_trick) == 4:
        winner = trick_winner(m.current_trick)
        m.scores[winner] += trick_points(m.current_trick)
        m.history.extend(m.current_trick)
        m.current_trick = []
        m.trick_number += 1
        m.to_play = winner
    else:
        m.to_play = (seat + 1) % 4
    return was_legal


def mirror_from_history(state, strict: bool = False):
    """Our GameState equal to the OpenSpiel state's determinized world.

    Returns (mirror, n_illegal_replays). `strict=True` asserts every recorded
    play is legal under our rules (true for the real world; used by the tests).
    """
    hist = state.history()
    hands = [0, 0, 0, 0]
    for i in range(DEAL_START, DEAL_END):
        hands[(i - DEAL_START) % 4] |= cards.bit(adapter.os_to_ours(hist[i]))
    m = GameState(hands=hands)
    for seat in range(4):
        if hands[seat] & cards.bit(cards.TWO_CLUBS):
            m.to_play = seat
    n_illegal = 0
    for a in hist[DEAL_END:]:
        card = adapter.os_to_ours(a)
        if strict:
            m.play(card)
        elif not _play_unchecked(m, card):
            n_illegal += 1
    return m, n_illegal


class HeuristicRolloutEvaluator(Evaluator):
    def __init__(self, prior_eps=None):
        assert prior_eps is None or 0.0 <= prior_eps <= 1.0, prior_eps
        self.prior_eps = prior_eps
        self._heuristic = HeuristicPlayer()
        self.n_evaluate = 0
        self.n_prior = 0
        self.n_inconsistent_worlds = 0   # evaluate() calls whose world contradicts a recorded play
        self.n_illegal_replays = 0       # total recorded plays illegal in their sampled world

    def evaluate(self, state):
        self.n_evaluate += 1
        m, n_illegal = mirror_from_history(state)
        if n_illegal:
            self.n_inconsistent_worlds += 1
            self.n_illegal_replays += n_illegal
        if not m.is_over():
            kernel.run_playout(m)
        return np.array([26.0 - s for s in m.scores], dtype=float)

    def prior(self, state):
        self.n_prior += 1
        if state.is_chance_node():
            return state.chance_outcomes()
        legal = list(state.legal_actions())
        n = len(legal)
        if self.prior_eps is None:
            return [(a, 1.0 / n) for a in legal]
        m, _ = mirror_from_history(state)
        view = m.view_for(m.to_play)
        a_h = adapter.ours_to_os(self._heuristic.choose(view))
        assert a_h in legal, (a_h, legal)
        eps = self.prior_eps
        return [(a, eps / n + (1.0 - eps if a == a_h else 0.0)) for a in legal]
