"""Choice-aware posterior: reweight candidate worlds by how well they explain
the opponents' ACTUAL plays.

The Phase-1 BeliefTable extracts everything the *constraints* say (who is void,
how many cards each seat holds). It provably tops out around 0.59 P(truth)
entering trick 13, because it never models HOW opponents choose. But a
deterministic opponent's every choice is a fingerprint of its hand: if the
heuristic discards its highest heart and that card was the 8H, the world in
which it also held the QH is impossible.

This module adds that evidence. For a candidate world (a concrete assignment of
the unseen cards to the three opponents) we reconstruct the full original deal
and replay the observed play sequence from the 2-clubs lead, asking at every
OPPONENT decision: "under this world, how likely was the policy to play the card
we actually saw?" The product of those likelihoods is the world's weight.
Weighted marginals over many candidate worlds give the posterior.

Honesty notes (all of these belong in any writeup that uses this class):

* Candidate worlds are drawn from the CONSTRAINT posterior (the Phase-1 belief
  table + sampler) and then reweighted by play likelihood. This is importance
  sampling: correct in the limit of infinitely many draws, but with finitely
  many draws the estimate inherits the sampler's known card-by-card bias AND
  adds weighting variance on top. `n_effective` = (sum w)^2 / sum w^2 is the
  health metric -- it says how many worlds really survived -- and every
  experiment that uses this class reports it.
* With epsilon = 0 the weights are 0/1, so this is exactly rejection sampling
  against the policy. With epsilon > 0 it is importance weighting.
* The epsilon term is uniform-over-ALL-legal-moves smoothing. That is a
  deliberately crude noise model: it is NOT the deviation law of
  `RandomizedHeuristic`, which excludes the heuristic's own choice and spreads
  epsilon over the other num_legal - 1 moves. Epsilon here is a robustness
  knob quantifying "my opponent model may be wrong", not a claim to model any
  particular deviant exactly.
* Truth-safety: a world whose replay is ILLEGAL (the observed card is not a
  legal move in that world) gets weight exactly 0.0 even when epsilon > 0.
  That does not violate the truth-safety rule: the TRUE world always replays
  legally, so it is never the world being killed. Only the policy factor is
  smoothed by epsilon. If every sampled world dies, that is a loud error, never
  a silent fallback.
"""
import numpy as np

from openhearts.belief.table import BeliefTable
from openhearts.engine import cards, kernel
from openhearts.engine.game import legal_moves
from openhearts.engine.state import GameState, PlayerView
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.sampler.sampler import sample_arrangement


def play_likelihood(policy, view: PlayerView, observed_card: int,
                    epsilon: float) -> float:
    """P(policy plays `observed_card` from `view`) under epsilon smoothing.

    (1 - epsilon) * [policy.choose(view) == observed_card] + epsilon / num_legal
    With epsilon = 0 this is exactly 1.0 or 0.0.
    """
    num_legal = len(cards.cards_in(view.legal_moves))
    assert num_legal > 0, "no legal moves at a decision point"
    hit = 1.0 if policy.choose(view) == observed_card else 0.0
    return (1.0 - epsilon) * hit + epsilon / num_legal


def _reconstruct_original_hands(view: PlayerView, world_hands):
    """Rebuild all four ORIGINAL 13-card hands from a candidate world.

    `world_hands[i]` is opponent i's CURRENT hand, indexed in
    `BeliefTable.opponent_seats` order, i.e. seats (view.seat + 1 + i) % 4.
    `view.hand` is the observer's CURRENT hand. Cards already played are gone
    from all of these, so we add each played card back to the seat that played
    it -- history plus the partial current trick.
    """
    observer = view.seat
    opponent_seats = [(observer + 1 + i) % 4 for i in range(3)]

    hands = [0, 0, 0, 0]
    hands[observer] = view.hand
    for i, s in enumerate(opponent_seats):
        hands[s] = int(world_hands[i])

    all_plays = list(view.history) + list(view.current_trick)
    for s, c in all_plays:
        b = cards.bit(c)
        assert not (hands[s] & b), (
            f"seat {s} still holds already-played card {c}"
        )
        hands[s] |= b

    # Structural asserts: separate "impossible world" from "broken
    # reconstruction". A misordered world_hands would otherwise surface only
    # as a silent zero weight further down.
    union = 0
    for s in range(4):
        assert bin(hands[s]).count("1") == 13, (
            f"reconstructed hand for seat {s} has "
            f"{bin(hands[s]).count('1')} cards, expected 13"
        )
        assert not (union & hands[s]), "reconstructed hands overlap"
        union |= hands[s]
    assert union == cards.FULL_DECK, "reconstructed deal is not a full deck"
    assert hands[observer] & view.hand == view.hand, (
        "observer's current hand is not contained in its reconstructed hand"
    )
    return hands, all_plays


def world_weight(view: PlayerView, world_hands, policy,
                 epsilon: float) -> float:
    """Likelihood that `policy` produced the observed plays in this world.

    `world_hands`: the three opponents' CURRENT hands, in
    `BeliefTable.opponent_seats` order (seats (view.seat+1+i) % 4).

    Replays the observed sequence from the 2-clubs lead on the reconstructed
    deal, multiplying in `play_likelihood` at every OPPONENT decision, computed
    on the replayed state's `view_for(that seat)`. The observer's own plays
    contribute 1.0 -- we know why we played them, they carry no information
    about the hidden hands. Returns 0.0 the moment the world cannot produce the
    observed play (illegal move, or a zero policy factor under epsilon = 0).

    Dispatch (Phase 3.5): when the JIT is active AND `policy` is exactly a
    plain `HeuristicPlayer`, the replay runs in `kernel.audit_world`, which is
    a verified port of the loop below (tests/test_jit_audit.py pins them
    bitwise). Any other policy -- including a subclass overriding `choose` --
    takes the Python path, because the kernel's policy is hardcoded. The
    Python path itself is untouched and remains the reference.
    """
    assert 0.0 <= epsilon <= 1.0, f"epsilon out of range: {epsilon}"
    hands, all_plays = _reconstruct_original_hands(view, world_hands)
    if not all_plays:
        return 1.0  # no evidence yet: every world is equally consistent
    _assert_two_clubs_leads(hands, all_plays)
    if kernel.jit_enabled() and type(policy) is HeuristicPlayer:
        return kernel.audit_world_weight(hands, all_plays, view.seat, epsilon)
    return _replay_weight_python(view, hands, all_plays, policy, epsilon)


def world_weight_python(view: PlayerView, world_hands, policy,
                        epsilon: float) -> float:
    """The pure-Python reference audit; never dispatches to the kernel."""
    assert 0.0 <= epsilon <= 1.0, f"epsilon out of range: {epsilon}"
    hands, all_plays = _reconstruct_original_hands(view, world_hands)
    if not all_plays:
        return 1.0
    _assert_two_clubs_leads(hands, all_plays)
    return _replay_weight_python(view, hands, all_plays, policy, epsilon)


def _assert_two_clubs_leads(hands, all_plays):
    # The replay must start at the 2-clubs holder, exactly as eval/guessing.py
    # does; starting at seat 0 would desync seats and score the wrong views.
    leader = next(s for s in range(4) if hands[s] & cards.bit(cards.TWO_CLUBS))
    assert leader == all_plays[0][0], (
        f"world's 2c holder {leader} did not lead the observed game "
        f"(observed leader {all_plays[0][0]})"
    )


def _replay_weight_python(view: PlayerView, hands, all_plays, policy,
                          epsilon: float) -> float:
    state = GameState(hands=hands)
    state.to_play = all_plays[0][0]

    observer = view.seat
    weight = 1.0
    for seat, card in all_plays:
        assert seat == state.to_play, (
            f"replay desync: observed seat {seat} != {state.to_play}"
        )
        legal = legal_moves(state.hands[seat], tuple(state.current_trick),
                            state.hearts_broken, state.trick_number)
        if not (legal & cards.bit(card)):
            return 0.0  # this world could not have produced the observed play
        if seat != observer:
            factor = play_likelihood(policy, state.view_for(seat), card,
                                     epsilon)
            assert np.isfinite(factor) and factor >= 0.0, (
                f"bad likelihood factor {factor}"
            )
            if factor == 0.0:
                return 0.0  # epsilon = 0 fast path: rejection
            weight *= factor
        state.play(card)

    assert np.isfinite(weight) and weight >= 0.0, f"bad world weight {weight}"
    return weight


class WeightedPosterior:
    """Weighted marginals P(opponent i holds card c) over surviving worlds.

    `probs` is (3, 52) and indexed by opponent index in `opponent_seats` order
    -- the same convention as `BeliefTable.probs`, so it is drop-in compatible
    with `eval.guessing.metrics_for`. Only CURRENT unseen cards can be nonzero.

    `unseen_mask` and `opponent_seats` are carried over from the proposal table
    so callers can build their truth mapping without rebuilding a table.
    """

    def __init__(self, probs, opponent_seats, unseen_mask, hand_sizes,
                 n_effective, draws_used, n_worlds_used, total_weight):
        self.probs = probs
        self.opponent_seats = opponent_seats
        self.unseen_mask = unseen_mask
        self.hand_sizes = hand_sizes
        self.n_effective = n_effective
        self.draws_used = draws_used
        self.n_worlds_used = n_worlds_used
        self.total_weight = total_weight

    @classmethod
    def from_view(cls, view: PlayerView, level, policy, epsilon: float,
                  n_worlds: int, rng, max_draws: int) -> "WeightedPosterior":
        """Draw candidate worlds from the constraint posterior and reweight.

        Proposal distribution: `BeliefTable.from_view(view, level)` plus the
        Phase-1 sampler. (The plan's prose says "FULL-level table"; the
        signature takes `level`, and `level` is what is used, so the caller
        chooses.) Draws until `n_worlds` worlds have positive weight or
        `max_draws` candidates have been drawn. Raises if nothing survives.

        On the JIT path candidates are drawn in batches through
        `kernel.sample_arrangements` and audited by `kernel.audit_world`.
        `draws_used`, `n_worlds_used` and `n_effective` mean exactly what
        they mean on the Python path (chunks are sized so the draw count is
        unchanged), but the random STREAM differs -- the batch sampler is
        seeded once per batch from `rng` instead of consuming it per card.
        Same caveat class as Phase 2.6; the Python path is bitwise unchanged.
        """
        table = BeliefTable.from_view(view, level)
        unseen = cards.cards_in(table.unseen_mask)

        probs = np.zeros((3, 52))
        total_w = 0.0
        total_w2 = 0.0
        draws = 0
        kept = 0

        if not unseen:  # nothing hidden: degenerate but well-defined
            return cls(probs, list(table.opponent_seats), table.unseen_mask,
                       list(table.hand_sizes), 0.0, 0, 0, 0.0)

        # Phase 3.5: on the JIT path draw candidates in batches through the
        # 2.6 batch sampler. Chunk size is min(remaining draws, worlds still
        # wanted), so a chunk can never be cut short by reaching n_worlds --
        # `draws` therefore counts exactly what the one-at-a-time loop below
        # would have counted. The draw STREAM differs (the batch sampler is
        # seeded once per call from `rng`); that is the same documented
        # caveat as Phase 2.6, and the Python path is bitwise unchanged.
        batched = kernel.jit_enabled() and type(policy) is HeuristicPlayer

        def account(world_hands, w):
            nonlocal kept, total_w, total_w2
            assert np.isfinite(w) and w >= 0.0, f"bad world weight {w}"
            if w <= 0.0:
                return
            kept += 1
            total_w += w
            total_w2 += w * w
            for i in range(3):
                for c in cards.cards_in(world_hands[i]):
                    probs[i, c] += w

        while kept < n_worlds and draws < max_draws:
            if batched:
                chunk = min(max_draws - draws, n_worlds - kept)
                batch, _n_failed = kernel.sample_arrangements(table, rng,
                                                              chunk)
                draws += chunk
                for world_hands in batch:
                    account(world_hands,
                            world_weight(view, world_hands, policy, epsilon))
                continue
            draws += 1
            drawn = sample_arrangement(table, rng)
            if drawn is None:
                continue  # sampler hit its restart cap; count the draw, move on
            world_hands, _attempts = drawn
            account(world_hands,
                    world_weight(view, world_hands, policy, epsilon))

        if total_w <= 0.0:
            raise AssertionError(
                f"no candidate world survived choice filtering after {draws} "
                f"draws (epsilon={epsilon}, level={level}); this is a loud "
                f"failure by design, never a silent fallback"
            )

        probs /= total_w
        assert np.isfinite(probs).all() and (probs >= 0.0).all()
        n_effective = (total_w * total_w) / total_w2
        return cls(probs, list(table.opponent_seats), table.unseen_mask,
                   list(table.hand_sizes), float(n_effective), draws, kept,
                   float(total_w))
