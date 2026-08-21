"""Bridges our engine to OpenSpiel's `hearts` game for C-bench 1.

The OpenSpiel `pyspiel.State` is the single source of truth that drives the
match and that the (unmodified, native) OpenSpiel ISMCTS bot sees. Our own
bot never touches that state: we maintain a harness-side mirror `GameState`
(same information-boundary pattern as eval/harness.py) that replays the
identical action sequence, and our bot is only ever handed a `PlayerView`
via `mirror.view_for(seat)`. See experiments/cbench/RULES_ALIGNMENT.md for
the full rules comparison and the card-id bijection derivation.
"""
import numpy as np

from openhearts.engine import cards
from openhearts.engine.game import deal as our_deal
from openhearts.engine.state import GameState

# qs_breaks_hearts=False to match our engine exactly: our hearts_broken flag
# is only set by playing a heart, never by Q(spades) alone (see
# RULES_ALIGNMENT.md sec 1). Confirmed empirically (2026-08-21): leaving
# OpenSpiel's default qs_breaks_hearts=True fires the legal-move tripwire
# whenever Q(spades) is played in a heart-less trick and someone next leads
# a heart our engine still considers "not broken."
GAME_STRING = "hearts(pass_cards=False,qs_breaks_hearts=False)"

# --- card id bijection -----------------------------------------------------
# OpenSpiel: card_os = rank*4 + suit_os, suit_os order C,D,H,S (0,1,2,3).
# Ours:      card    = suit_ours*13 + rank, suit_ours order C,D,S,H (0,1,2,3).
_SUIT_OS_TO_OURS = {0: 0, 1: 1, 2: 3, 3: 2}
_SUIT_OURS_TO_OS = {v: k for k, v in _SUIT_OS_TO_OURS.items()}


def os_to_ours(card_os: int) -> int:
    card_os = int(card_os)  # ISMCTS/numpy may hand back numpy ints; bitmask
    suit_os, rank = card_os % 4, card_os // 4  # ops on those poison hands.
    return int(_SUIT_OS_TO_OURS[suit_os] * 13 + rank)


def ours_to_os(card_ours: int) -> int:
    card_ours = int(card_ours)
    suit_ours, rank = card_ours // 13, card_ours % 13
    return int(rank * 4 + _SUIT_OURS_TO_OS[suit_ours])


# --- deal control ------------------------------------------------------------

def force_deal(state, seed: int):
    """Deal via our own `deal()` (seeded), then push the identical cards into
    an OpenSpiel state via its chance nodes. Returns the resulting GameState
    (unplayed, all 4 hands known) for use as the mirror's starting point.

    `state` must be a fresh `pyspiel` state (`is_chance_node()` for the pass
    direction choice, not yet dealt).
    """
    rng = np.random.default_rng(seed)
    our_state = our_deal(rng)

    # Pass-direction chance node. OpenSpiel always offers 4 uniform outcomes
    # here (its ChanceOutcomes() doesn't special-case pass_cards=False), but
    # with pass_cards=False no pass phase ever runs, so the direction picked
    # is inert -- any of the 4 is equivalent. We pick action 0 ("No Pass",
    # per action_to_string) deterministically for reproducibility.
    assert state.is_chance_node()
    outcomes = state.chance_outcomes()
    assert len(outcomes) == 4, f"unexpected pass-dir outcomes: {outcomes}"
    state.apply_action(0)

    # OpenSpiel's dealing recipient is fixed by the dealt-so-far count
    # (`holder_[card] = num_cards_dealt_ % kNumPlayers`, hearts.cc) -- WHO
    # gets the next card is not our choice, only WHICH card is. So we must
    # interleave round-robin (one card per seat per round, 13 rounds) rather
    # than dealing seat 0's whole hand, then seat 1's, etc. The physical
    # order within a seat's 13 cards doesn't matter, only the final holding.
    per_seat_cards = [cards.cards_in(our_state.hands[seat]) for seat in range(4)]
    for round_idx in range(13):
        for seat in range(4):
            card_ours = per_seat_cards[seat][round_idx]
            card_os = ours_to_os(card_ours)
            assert state.is_chance_node()
            legal_os_ids = {a for a, _ in state.chance_outcomes()}
            assert card_os in legal_os_ids, (
                f"card {card_ours} (os {card_os}) not a legal deal outcome; "
                f"legal os ids: {sorted(legal_os_ids)}"
            )
            state.apply_action(card_os)

    assert not state.is_chance_node()
    return our_state


# --- legality tripwire -------------------------------------------------------

def assert_legal_agreement(os_state, mirror: GameState):
    """Abort loudly if the mirror's legal moves (our engine) disagree with
    OpenSpiel's own legal_actions() at the current decision point. This is
    the live rules-alignment tripwire described in RULES_ALIGNMENT.md.
    """
    seat = mirror.to_play
    assert os_state.current_player() == seat, (
        f"seat mismatch: openspiel current_player={os_state.current_player()} "
        f"mirror.to_play={seat}"
    )
    view = mirror.view_for(seat)
    ours_legal = set(cards.cards_in(view.legal_moves))
    os_legal = {os_to_ours(a) for a in os_state.legal_actions()}
    if ours_legal != os_legal:
        raise AssertionError(
            "LEGAL-MOVE TRIPWIRE FIRED: rules disagreement between our "
            f"engine and OpenSpiel at seat {seat}, trick {mirror.trick_number}.\n"
            f"  ours (translated to our ids):      {sorted(ours_legal)}\n"
            f"  openspiel (translated to our ids): {sorted(os_legal)}\n"
            f"  mirror.current_trick={mirror.current_trick} "
            f"hearts_broken={mirror.hearts_broken}"
        )


def apply_both(os_state, mirror: GameState, card_ours: int):
    """Play `card_ours` (our card id) on both state machines in lockstep."""
    card_os = ours_to_os(card_ours)
    os_state.apply_action(card_os)
    mirror.play(card_ours)


def rescore(mirror: GameState):
    """Our own points tally (no moon-shoot rule) -- see RULES_ALIGNMENT.md
    §3. This is authoritative for C-bench; OpenSpiel's own state.returns()
    is not used for the reported match result."""
    return list(mirror.scores)
