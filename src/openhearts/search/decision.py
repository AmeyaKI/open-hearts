"""Move choice by sampled playouts ("determinization").

For each decision we build a belief table from the view, sample several
concrete guesses at the opponents' hands, and for every legal move play the
hand out to the end with heuristic players in all four seats. The move with
the lowest average points-from-here wins.

Known limitations (also in the README, deliberately not fixed in Phase 1):
inside an imagined arrangement the playout players effectively know every
card, so the search assumes opponents are better informed than they really
are and undervalues plays that create uncertainty. Playouts also use the very
same heuristic as the real opponents -- a perfect opponent model, which
flatters the bot relative to real-world play.
"""
from openhearts.belief.table import BeliefTable, Level
from openhearts.engine import cards, kernel
from openhearts.engine.state import GameState, PlayerView
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.sampler.sampler import sample_arrangement


def state_from_view(view: PlayerView, sampled_hands) -> GameState:
    """Rebuild a full imagined GameState from a filtered view.

    `sampled_hands[i]` is the guessed hand of seat `(view.seat + 1 + i) % 4`,
    matching `BeliefTable.opponent_seats` order. Everything the view carries
    is copied across as mutable containers, because GameState.play mutates
    them in place.
    """
    hands = [0, 0, 0, 0]
    hands[view.seat] = view.hand
    for i in range(3):
        hands[(view.seat + 1 + i) % 4] = sampled_hands[i]
    return GameState(
        hands=hands,
        history=list(view.history),
        current_trick=list(view.current_trick),
        hearts_broken=view.hearts_broken,
        trick_number=view.trick_number,
        scores=list(view.scores),
        to_play=view.seat,
    )


class SearchPlayer:
    def __init__(self, level: Level, n_samples: int, rng,
                 sampler_respects_voids: bool = True):
        # sampler_respects_voids=False exists for one experiment only: the
        # sampler normally refuses to deal a card into a suit its holder has
        # shown void in, which leaks void evidence into EVERY belief level --
        # including UNIFORM, whose table ignores voids. Turning it off gives a
        # truly uninformed control for the ablation.
        self.level = level
        self.n_samples = n_samples
        self.rng = rng
        self.sampler_respects_voids = sampler_respects_voids
        self.fallbacks = 0          # decisions that fell back to the heuristic
        self.failed_samples = 0     # arrangements the sampler could not build
        self._heuristic = HeuristicPlayer()

    def choose(self, view: PlayerView) -> int:
        legal = cards.cards_in(view.legal_moves)
        if len(legal) == 1:
            return legal[0]

        table = BeliefTable.from_view(view, self.level)
        if not self.sampler_respects_voids:
            table = BeliefTable(table.probs, [set(), set(), set()],
                                table.hand_sizes, table.opponent_seats,
                                table.unseen_mask)
        arrangements = []
        for _ in range(self.n_samples):
            result = sample_arrangement(table, self.rng)
            if result is None:
                self.failed_samples += 1
                continue
            hands, _attempts = result
            arrangements.append(hands)

        if len(arrangements) * 2 < self.n_samples:
            self.fallbacks += 1
            return self._heuristic.choose(view)

        base_score = view.scores[view.seat]
        best_card, best_avg = None, None
        for card in legal:
            total = 0
            for hands in arrangements:
                state = state_from_view(view, hands)
                state.play(card)
                self._playout(state)
                total += state.scores[view.seat] - base_score
            avg = total / len(arrangements)
            if best_avg is None or avg < best_avg:
                best_card, best_avg = card, avg
        return best_card

    def _playout(self, state: GameState) -> None:
        if kernel.jit_enabled():
            kernel.run_playout(state)
            return
        self._playout_python(state)

    def _playout_python(self, state: GameState) -> None:
        # Reference implementation; the numba kernel is an exact port of it
        # (tests/test_kernel_equivalence.py pins the two together). Kept as
        # live code so OPENHEARTS_NO_JIT=1 has something to run.
        players = [self._heuristic] * 4
        while not state.is_over():
            seat = state.to_play
            state.play(players[seat].choose(state.view_for(seat)))
