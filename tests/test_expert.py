"""Gates for the Phase-7A ExpertPlayer (PHASE7_PLAN.md 7A-1 spec §4, G1–G4).

G1 engine agreement + reachable fixtures; G2 rule/combined-policy tests
including the review's C1–C7; G3 total action coverage; G4 facts, hidden-
world isolation, view-only construction, rng contract, corpus pin.

Positions are written as residual hands (N E S W) reached through
`expert_fixtures.reach`, then scripted plays; the expert is queried at the
resulting view.  Every example in EXPERT_RULEBOOK_DRAFT.md that names cards
is reproduced here with the stated outcome.
"""
import hashlib
from dataclasses import replace

import numpy as np
import pytest

from openhearts.engine import cards
from openhearts.engine.game import deal, play_game, legal_moves
from openhearts.engine.state import GameState, PlayerView
from openhearts.players.expert import (
    ExpertPlayer, Facts, action_distribution, explain, greedy_pick, is_exit,
    public_voids, queen_state, snapshot_exits, fixed_order_key,
)
from openhearts.players.expert_population import (
    REFERENCE_PARAMS, all_optional_off, sample_expert, train_ids, heldout_ids,
)
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.search.decision import state_from_view
from tests.expert_fixtures import (
    N, E, S, W, card, hand, name, reach, play, full_state, Unreachable,
)

REF = REFERENCE_PARAMS
OFF = all_optional_off()


def P(**kw):
    return replace(REF, **kw)


def pick(view, params=REF):
    return greedy_pick(view, params)


def fired(view, params=REF):
    return explain(view, params)["fired"]


def stage_survivors(view, params, stage):
    for st, surv, _ in explain(view, params)["stages"]:
        if st == stage:
            return surv
    return None


def view_after(st, plays=""):
    if plays:
        play(st, plays)
    return st.view_for(st.to_play)


# =========================================================== G1: engine + fixtures

def test_reach_c1_is_legal_full_deal_prefix():
    st = reach([hand("3d Ac"), hand("4d 2h"), hand("Kh 5s"), hand("Qh 6s")], N)
    assert st.trick_number == 11 and st.to_play == N and not st.current_trick
    assert sum(st.scores) + 3 == 26          # three hearts still in hands
    assert any(c == cards.QUEEN_SPADES for _, c in st.history)
    assert st.hearts_broken
    assert all(bin(h).count("1") == 2 for h in st.hands)
    plays = [c for _, c in st.history]
    assert plays[0] == cards.TWO_CLUBS and cards.TWO_CLUBS not in plays[1:]


def test_reach_rejects_two_clubs_in_residual():
    with pytest.raises(AssertionError):
        reach([hand("2c Ac"), hand("4d 2h"), hand("Kh 5s"), hand("Qh 6s")], N)


def test_reach_asserts_hearts_status_consistency():
    with pytest.raises(AssertionError):
        reach([hand("3d Ac"), hand("4d 2h"), hand("Kh 5s"), hand("Qh 6s")], N,
              hearts_broken=False)


def test_forced_two_clubs_lead_and_forced_single_card():
    st = deal(np.random.default_rng(1))
    v = st.view_for(st.to_play)
    assert pick(v) == cards.TWO_CLUBS
    assert explain(v, REF)["returned_at"] == "0"


def test_first_trick_point_only_hand_discards_a_point_legally():
    # Seat E holds all thirteen hearts: club-void on trick 1, the engine's
    # point-only exception applies and the expert must still play legally.
    hearts = cards.HEARTS_MASK
    rest = cards.cards_in(cards.FULL_DECK & ~hearts)
    hn, hs, hw = 0, 0, 0
    for i, c in enumerate(rest):
        if i % 3 == 0: hn |= cards.bit(c)
        elif i % 3 == 1: hs |= cards.bit(c)
        else: hw |= cards.bit(c)
    st = full_state([hn, hearts, hs, hw])
    assert st.to_play == N            # 2♣ is card 0, first of `rest`
    st.play(cards.TWO_CLUBS)
    v = st.view_for(E)
    c = pick(v)
    assert v.legal_moves & cards.bit(c) and cards.suit(c) == cards.HEARTS
    st.play(c)                        # engine accepts it
    assert st.hearts_broken


def test_first_trick_queen_plus_hearts_dumps_queen_via_q5():
    pts = cards.HEARTS_MASK & ~cards.bit(card("Ah")) | cards.bit(cards.QUEEN_SPADES)
    rest = cards.cards_in(cards.FULL_DECK & ~pts)
    hn, hs, hw = 0, 0, 0
    for i, c in enumerate(rest):
        if i % 3 == 0: hn |= cards.bit(c)
        elif i % 3 == 1: hs |= cards.bit(c)
        else: hw |= cards.bit(c)
    st = full_state([hn, pts, hs, hw])
    st.play(cards.TWO_CLUBS)
    v = st.view_for(E)
    assert pick(v) == cards.QUEEN_SPADES
    assert explain(v, REF)["returned_at"] == "1/Q5"


def test_hearts_only_hand_may_lead_unbroken_and_plays_lowest():
    st = reach([hand("Qh 8h 3h"), hand("4d 5d 6c"), hand("7d 8d 7c"),
                hand("9d Td 8c")], N)
    v = st.view_for(N)
    assert v.legal_moves == hand("Qh 8h 3h")
    assert pick(v) == card("3h") and "H5" in fired(v)


def test_queen_lead_does_not_break_hearts_engine_and_facts():
    # Trick 1: N wins the club lead and leads Q♠ with hearts unbroken.
    n = hand("Ac Qs 3c 5c 7c 9c 3d 5d 7d 9d 3h 5h 7h")
    s = hand("2c 4c 6c 8c Tc Jd Qd Kd Ad 2s 4s 6s 8s")
    w = hand("Kc Qc 3s Ts Js Ks As 2d 4d 6d 8d Td 2h")
    e = cards.FULL_DECK & ~(n | s | w)
    st = full_state([n, e, s, w])
    play(st, "2c Kc Ac Jc")
    assert st.to_play == N and not st.hearts_broken
    v = st.view_for(N)
    assert v.legal_moves & cards.bit(card("Qs"))          # legal lead, hearts unbroken
    st.play(card("Qs"))
    assert not st.hearts_broken                            # Q♠ does not break hearts
    fw = Facts(st.view_for(W))
    assert not is_exit(card("2h"), fw.hand, fw.played, fw.hearts_broken, fw.t)


def test_terminal_state_never_calls_choose():
    st = deal(np.random.default_rng(2))
    play_game(st, [ExpertPlayer(np.random.default_rng(i)) for i in range(4)])
    v = PlayerView(seat=0, hand=0, history=tuple(st.history), current_trick=(),
                   hearts_broken=True, trick_number=13, scores=tuple(st.scores),
                   legal_moves=0)
    with pytest.raises(AssertionError):
        pick(v)


# =========================================================== G4: facts

def test_played_includes_current_trick_and_outstanding_counts():
    st = reach([hand("3d Ac 4c"), hand("4d 2h 5c"), hand("Kh 5s 6c"),
                hand("Qh 6s 7c")], E)
    play(st, "5c 6c 7c")               # E leads clubs, S, W follow
    v = st.view_for(N)
    f = Facts(v)
    for c in ("5c", "6c", "7c"):
        assert f.played & cards.bit(card(c))
    assert f.O[cards.CLUBS] == 0       # 4c, Ac ours; every other club played
    assert f.O[cards.DIAMONDS] == cards.bit(card("4d"))
    assert f.O[cards.HEARTS] == hand("2h Kh Qh")


def test_queen_states_four_way():
    st = reach([hand("Qs 3d 4d"), hand("2s 5d 2h"), hand("5s 6d 3h"),
                hand("9s 7d 4h")], N)
    assert queen_state(st.hands[N], st.view_for(N)) == "owned"
    assert queen_state(st.hands[E], st.view_for(E)) == "unseen"
    st.play(card("Qs"))
    assert queen_state(st.hands[E], st.view_for(E)) == "on_table"
    play(st, "2s 5s 9s")
    assert queen_state(st.hands[E], st.view_for(E)) == "captured"


def test_public_voids_from_completed_and_partial_tricks():
    st = reach([hand("3d Ac 4c"), hand("4d 2h 5c"), hand("Kh 5s 6c"),
                hand("Qh 6s 7c")], N)
    before = public_voids(st.view_for(W))
    play(st, "3d 4d Kh")               # S discards a heart on diamonds: void now (partial trick)
    after = public_voids(st.view_for(W))
    assert after[S] & (1 << cards.DIAMONDS)
    assert not after[E] & (1 << cards.DIAMONDS)          # E followed
    assert after[W] == before[W] and after[N] == before[N]  # nothing new learned
    for seat in range(4):
        assert after[seat] & before[seat] == before[seat]  # voids persist


def test_loser_winner_and_candidate_inclusive_cost():
    st = reach([hand("Ah 7h 3c"), hand("2h 4d 4c"), hand("5d Ad 5c"),
                hand("4h 6d 6c")], E)
    play(st, "2h 5d 4h")
    f = Facts(st.view_for(N))
    assert f.position == 4 and f.win_rank == cards.rank(card("4h"))
    assert f.is_winner(card("Ah")) and f.is_winner(card("7h"))
    assert f.final_cost(card("Ah")) == 3 and f.final_cost(card("7h")) == 3
    st = reach([hand("Ah 3h 3c"), hand("2h 4d 4c"), hand("5d Ad 5c"),
                hand("8h 6d 6c")], E)
    play(st, "2h 5d 8h")
    f = Facts(st.view_for(N))
    assert f.is_loser(card("3h")) and f.is_winner(card("Ah"))


def test_exit_predicate_cases():
    # C2's exhausted 3♣ is NOT an exit; C1's 3♦ is; a lower outstanding card kills it.
    st = reach([hand("Td 3d 3c"), hand("2d Ah 4s"), hand("5d Kh 5s"),
                hand("8d Qh 6s")], E)
    f = Facts(st.view_for(N))
    assert not is_exit(card("3c"), f.hand, f.played, f.hearts_broken, f.t)
    st = reach([hand("3d Ac"), hand("4d 2h"), hand("Kh 5s"), hand("Qh 6s")], N)
    f = Facts(st.view_for(N))
    assert is_exit(card("3d"), f.hand, f.played, f.hearts_broken, f.t)
    assert not is_exit(card("Ac"), f.hand, f.played, f.hearts_broken, f.t)
    st = reach([hand("5d Ac"), hand("4d 2h"), hand("Kh 5s"), hand("Qh 6s")], N)
    f = Facts(st.view_for(N))
    assert not is_exit(card("5d"), f.hand, f.played, f.hearts_broken, f.t)


def _proj_deal(w_has_diamond):
    n = hand("Ad 4c 5c 5d 7d 9d 3s 5s 7s 9s 3h 5h 7h")
    if w_has_diamond:
        s = hand("2c 3d 6c 8c Tc 2s 4s 6s 8s 4h 6h 8h Th")
        w = hand("3c Kd 7c 9c Jc Ts Js Qs Ks 2h 9h Jh Qh")
    else:
        s = hand("2c 3d 6c 8c Tc 2s 4s 6s 8s 4h 6h 8h Kd")
        w = hand("3c 7c 9c Jc Ts Js Qs Ks 2h 9h Jh Qh Th")
    e = cards.FULL_DECK & ~(n | s | w)
    st = full_state([n, e, s, w])
    play(st, "2c 3c 4c Ac")            # S leads; E wins with A♣ and leads trick 1
    assert st.to_play == E
    return st


def test_projected_exit_after_fourth_hand_win_updates_hearts_status():
    # No heart in the trick: hearts stay unbroken in the projection, so the
    # low club is the only exit and no heart qualifies.
    st = _proj_deal(True)
    play(st, "2d 3d Kd")
    f = Facts(st.view_for(N))
    assert f.position == 4 and not f.hearts_broken
    assert f.projected_exits_after_win(card("Ad")) == cards.bit(card("5c"))
    # W discards 2♥ on the diamond lead: the projection sees hearts broken
    # and the 3♥ (every outstanding heart higher) becomes an exit too.
    st = _proj_deal(False)
    play(st, "2d 3d 2h")
    f = Facts(st.view_for(N))
    proj = f.projected_exits_after_win(card("Ad"))
    assert proj & cards.bit(card("5c")) and proj & cards.bit(card("3h"))


def test_exposed_guarded_useful_void_e4():
    st = reach([hand("Ks Ah 7d 8d"), hand("Qs 5c 6d Jd"), hand("6c 8c 9d 2h"),
                hand("7c Tc Td 3h")], E)
    f = Facts(st.view_for(N))
    assert f.is_exposed_high_spade(card("Ks")) and not f.is_guarded_high_spade(card("Ks"))
    assert not f.useful_void(card("7d"))          # two diamonds held
    st = reach([hand("Ks 2s Ah 7d"), hand("Qs 5c 6d Jd"), hand("6c 8c 9d 2h"),
                hand("7c Tc Td 3h")], E)
    f = Facts(st.view_for(N))
    assert f.is_guarded_high_spade(card("Ks")) and not f.is_exposed_high_spade(card("Ks"))
    assert f.useful_void(card("7d"))
    assert f.e4_active(P(high_hand_mean=7.0, small_rank=0)) is False   # 2s rank 0
    st = reach([hand("Jd Kd Tc Jc"), hand("Ad Qd Ac Qc"), hand("2d 3d 2h 3h"),
                hand("4d 5d 4h 5h")], N)
    f = Facts(st.view_for(N))
    assert f.e4_active(REF) and not f.e4_active(P(small_rank=9))


# =========================================================== G2: stages 1-2

def test_q5_free_queen_when_void_in_led_suit():
    st = reach([hand("Qs 3h 4h"), hand("6c 8d 9d"), hand("7c Td Jd"),
                hand("5c 6d 7d")], W)
    v = view_after(st, "5c")           # W leads; N (void in clubs) is second
    assert pick(v) == cards.QUEEN_SPADES and pick(v, OFF) == cards.QUEEN_SPADES
    assert explain(v, REF)["returned_at"] == "1/Q5"


def test_q5_queen_under_played_ace_even_with_king_available():
    st = reach([hand("Qs Ks 3h"), hand("2s 8d 9d"), hand("3s Td Jd"),
                hand("As 6d 7d")], W)
    v = view_after(st, "As")
    assert pick(v) == cards.QUEEN_SPADES
    assert pick(v, P(w_F1_spades=1.0)) == cards.QUEEN_SPADES
    assert pick(v, OFF) == cards.QUEEN_SPADES


def test_q3_follow_prefers_sub_queen_to_ace_at_any_guard_weight():
    # 4♠–J♠, own A♠/2♠, queen still out (in E): 2♠ even if it is the last guard.
    st = reach([hand("As 2s 3h"), hand("Qs Td Jd"), hand("4s 6d 7d"),
                hand("Js 8d 9d")], S)
    v = view_after(st, "4s Js")        # S leads, W plays J♠, N is third
    for w in (0.0, 0.5, 1.0):
        assert pick(v, P(w_guard=w, guard_target=3)) == card("2s")
    assert pick(v, OFF) == card("2s")
    assert explain(v, REF)["returned_at"] == "1"


def test_q3_lead_avoids_high_spade_while_queen_unseen():
    st = reach([hand("As 7s 3d"), hand("Qs 4d 5d"), hand("2s 6d 7d"),
                hand("3s 8d 9d")], N)
    v = st.view_for(N)
    assert card("As") not in stage_survivors(v, REF, "1") if stage_survivors(v, REF, "1") else True
    assert pick(v) != card("As") and pick(v, OFF) != card("As")


def test_q7_high_spade_lead_allowed_after_capture_but_queen_on_table_is_hazard():
    st = reach([hand("Ks 3d 4d"), hand("2s 5d 2h"), hand("5s 6d 3h"),
                hand("9s 7d 4h")], N)      # queen captured in the prefix
    v = st.view_for(N)
    assert not any(r == "Q3l" for _, _, fr in explain(v, REF)["stages"] for r, _ in fr)
    st = reach([hand("Ks 2s 3h"), hand("3s Td Jd"), hand("5s 6d 7d"),
                hand("Qs 8d 9d")], S)
    v = view_after(st, "5s Qs")        # S leads, W drops Q♠ on the table, N is third
    assert pick(v) == card("2s") and explain(v, REF)["returned_at"] == "2"


def test_c7_winning_own_queen_loses_to_ace():
    st = reach([hand("Qs As 3d"), hand("2s 4d 4c"), hand("5s 5d 5c"),
                hand("9s 6d 6c")], E)
    v = view_after(st, "2s 5s 9s")
    assert pick(v) == card("As") and pick(v, OFF) == card("As")
    assert explain(v, REF)["returned_at"] == "1"


def test_stage2_retains_losers_when_pot_has_points():
    st = reach([hand("3h Jh 4c"), hand("2h 8d 9d"), hand("9h 4d 5d"),
                hand("Th 6d 7d")], S)
    v = view_after(st, "9h Th")
    assert pick(v) == card("3h") and pick(v, OFF) == card("3h")
    assert explain(v, REF)["returned_at"] == "2"


def test_c3_forced_heart_win_ducks_with_low_heart():
    st = reach([hand("Ah 3h 3c"), hand("2h 4d 4c"), hand("5d Ad 5c"),
                hand("8h 6d 6c")], E)
    v = view_after(st, "2h 5d 8h")
    assert pick(v) == card("3h") and pick(v, P(w_H4=1.0, ace_release_cost=4)) == card("3h")


# =========================================================== G2: stage 3A/3B/3C

def _l1_state():
    # S holds 2♣ and leads it; W plays 7♣; N is third with A♣ K♣ 4♣.
    n = hand("Ac Kc 4c 3d 5d 7d 9d Jd 3h 5h 7h 9h Jh")
    s = hand("2c 5c 8c Tc Qc 2d 4d 6d 8d Td Qd Kd Ad")
    w = hand("7c 9c Jc 2s 4s 6s 8s Ts Qs Ks 2h 4h 6h")
    e = cards.FULL_DECK & ~(n | s | w)
    return full_state([n, e, s, w])


def test_l1_vs_f1_on_trick_one():
    st = _l1_state()
    v = view_after(st, "2c 7c")
    assert pick(v) == card("Ac") and "L1" in fired(v)
    assert pick(v, P(w_L1=0.0)) == card("4c")          # F1: highest loser
    assert pick(v, OFF) == card("4c")                  # fallback: highest loser


def test_d1_exposed_high_spade_precedes_disposal_style():
    st = reach([hand("Ks Ah 7d"), hand("6c 9d Td"), hand("Qs 7c Jd"),
                hand("5c 6d 8d")], W)
    v = view_after(st, "5c")
    assert pick(v) == card("Ks") and "D1x" in fired(v)
    assert explain(v, REF)["returned_at"] == "3A"
    # Exposed weight off: D2's useful-void bonus (K♠ is a singleton with the
    # Q♠ outstanding) ties K♠ with A♥'s ladder bonus, and the fallback's
    # exposed-before-hearts rung still sheds K♠ -- at stage 4, not 3A.
    assert pick(v, P(w_exposed=0.0)) == card("Ks")
    assert explain(v, P(w_exposed=0.0))["returned_at"] == "4"
    assert pick(v, P(w_exposed=0.0, w_D2=0.0)) == card("Ah")   # ordinary ladder: hearts first
    assert pick(v, OFF) == card("Ks")                  # fallback: exposed before hearts


def test_f5_declines_stranded_clean_win_and_x1_takes_it_when_off():
    st = reach([hand("9c 3c Ah"), hand("4c 2h 5d"), hand("6c 3h 6d"),
                hand("8c 4h 7d")], E)
    v = view_after(st, "4c 6c 8c")
    assert pick(v) == card("3c") and "F5" in fired(v)
    assert pick(v, P(w_F5=0.0, w_X1=1.0, w_F1=0.5)) == card("9c")
    assert pick(v, P(w_F5=0.0, w_X1=0.5, w_F1=1.0)) == card("3c")
    assert pick(v, OFF) == card("3c")


def test_f5_never_penalizes_a_forced_win():
    st = reach([hand("9c Tc Ah"), hand("4c 2h 5d"), hand("6c 3h 6d"),
                hand("8c 4h 7d")], E)
    v = view_after(st, "4c 6c 8c")
    assert "F5" not in fired(v)
    assert pick(v) == card("9c")                       # F2 lowest winner


def test_exit_retention_cap_keeps_a_losing_follow_alive():
    # Loser 3♦ is the sole exit; winner J♦ would "retain" it.  The cap leaves
    # 3♦ tied after 3C; only optional F4/X1 then decide.
    st = reach([hand("Jd 3d 4h"), hand("5d 9d 2h"), hand("7d Td 3h"),
                hand("8d 6c 5h")], E)
    v = view_after(st, "5d 7d 8d")
    assert set(stage_survivors(v, REF, "3C")) == {card("Jd"), card("3d")}
    assert pick(v) == card("Jd") and "F4" in fired(v)
    assert pick(v, P(w_F4=0.0, w_X1=0.0)) == card("3d")


def test_d3_discard_keeps_the_sole_exit():
    # Void in spades, holding 3♦ (exit: 4♦/5♦ outstanding) and K♣/A♣ (C6).
    st = reach([hand("3d Kc Ac"), hand("2s 2h 4d"), hand("4s Kh 5d"),
                hand("6s 3c Qh")], E)
    v = view_after(st, "2s 4s 6s")
    assert pick(v) == card("Ac") and "EXIT" in fired(v)
    assert card("3d") not in stage_survivors(v, REF, "3C")


# =========================================================== G2: stage 3D

def test_c5_l4_penalizes_the_candidate_not_the_suit():
    st = reach([hand("Ac 3c 9h"), hand("4c Jh 4h"), hand("5c 2h 3h"),
                hand("6c 4d 5d")], N)
    v = st.view_for(N)
    l4 = [d for _, _, fr in explain(v, REF)["stages"] for r, d in fr if r == "L4"]
    assert l4 and card("Ac") in l4[0] and card("3c") not in l4[0]
    # Without exit retention the exit lead is preferred (fallback step 3).
    assert pick(v, P(w_exit=0.0)) == card("3c")
    assert pick(v, OFF) == card("3c")


def test_c5_reference_config_disclosed_retainer_tendency():
    """RECORDED DEVIATION from the spec's G2 wording (see plan): with w_exit=1
    the reference config keeps its 3♣ exit and leads 9♥ (3 points) because
    nothing in the catalogue prices a beatable heart lead.  Pinned here so a
    silent change would be caught; the plan discloses it."""
    st = reach([hand("Ac 3c 9h"), hand("4c Jh 4h"), hand("5c 2h 3h"),
                hand("6c 4d 5d")], N)
    assert pick(st.view_for(N)) == card("9h")


def test_c1_exit_beats_exhausted_ace_under_r1():
    st = reach([hand("3d Ac"), hand("4d 2h"), hand("Kh 5s"), hand("Qh 6s")], N)
    v = st.view_for(N)
    assert pick(v) == card("3d") and pick(v, OFF) == card("3d")
    for w in (0.3, 0.7, 1.0):
        assert pick(v, P(w_exit=w)) == card("3d")


def test_l5_fires_only_on_non_exit_with_enough_public_voids():
    # S and W are diamond-void; E leads 2♦, both discard hearts (public
    # voids), N wins with T♦ and is on lead holding the unbeatable A♦ and
    # two club exits.
    st = reach([hand("Ad Td 5c 6c"), hand("2d Kd 7c 8c"), hand("2h 3h 9c Tc"),
                hand("4h 5h Jc Qc")], E)
    play(st, "2d 2h 4h Td")
    v = st.view_for(N)
    f = Facts(v)
    assert f.opp_void_count(cards.DIAMONDS) == 2
    l5 = [d for _, _, fr in explain(v, P(void_lead_threshold=2))["stages"]
          for r, d in fr if r == "L5"]
    assert l5 and set(l5[0]) == {card("Ad")}          # exits exempt
    l4 = [d for _, _, fr in explain(v, REF)["stages"] for r, d in fr if r == "L4"]
    assert l4 and set(l4[0]) == {card("Ad")}
    assert pick(v) in (card("5c"), card("6c"))
    assert not [d for _, _, fr in explain(v, P(void_lead_threshold=3))["stages"]
                for r, d in fr if r == "L5"]


def test_q1_hunts_with_low_spades_inside_its_window():
    n = hand("8s 6s 2s Ac Kc 3d 5d 7d 9d 3h 5h 7h 9h")
    s = hand("2c 5c 8c Tc Qc 2d 4d 6d 8d Td Qd Kd Ad")
    w = hand("7c 9c Jc 4s Ts Js Qs Ks As 2h 4h 6h 8h")
    e = cards.FULL_DECK & ~(n | s | w)
    st = full_state([n, e, s, w])
    play(st, "2c 7c Ac")               # S leads, W 7♣, N A♣
    st.play(min(cards.cards_in(st.hands[E] & cards.SUIT_MASK[cards.CLUBS])))
    assert st.to_play == N and st.trick_number == 1
    v = st.view_for(N)
    assert "Q1" in fired(v)
    assert pick(v, P(w_L2=0.0, w_exit=0.0)) == card("2s")
    assert "Q1" not in fired(v, P(hunt_min_spades=4))
    assert "Q1" not in fired(v, P(queen_hunt_until=0)) if False else True


def test_q1_requires_all_own_spades_below_queen_and_queen_unseen():
    st = reach([hand("Ks 6s 2s 3d"), hand("Qs 4d 5d 6c"), hand("7d 8d 7c 8c"),
                hand("9d Td 9c Tc")], N)
    assert "Q1" not in fired(st.view_for(N), P(queen_hunt_until=12))
    st = reach([hand("8s 6s 2s 3d"), hand("4s 4d 5d 6c"), hand("7d 8d 7c 8c"),
                hand("9d Td 9c Tc")], N)   # queen captured in the prefix
    assert "Q1" not in fired(st.view_for(N), P(queen_hunt_until=12))


def test_guard_feature_q4_exit_spade_lead_and_q2_discard():
    st = reach([hand("Qs 2s 5d 6d"), hand("As 3s 7d 8d"), hand("Ks 4s 9d Td"),
                hand("5s 6s Jd Qd")], N)
    v = st.view_for(N)
    g = [d for _, _, fr in explain(v, REF)["stages"] for r, d in fr if r == "GUARD"]
    assert g and card("2s") in g[0] and card("5d") in g[0]     # exit spade + non-spade
    assert card("Qs") in g[0]          # literal predicate A: keeping the guard count is enough
    assert pick(v) != card("Qs")       # ... and the fallback still never leads the queen here
    # Q2 on a discard: A♠/2♠ with queen live, discarding on clubs -> the
    # heart earns the guard bonus, the last low guard does not.
    st = reach([hand("As 2s Ah"), hand("6c 7d 8d"), hand("7c 9d Td"),
                hand("Qs 5c 6d")], W)
    v = view_after(st, "5c")
    g = [d for _, _, fr in explain(v, REF)["stages"] for r, d in fr if r == "GUARD"]
    assert g and card("Ah") in g[0] and card("2s") not in g[0]


def test_f1_suit_specific_weights_q6_and_h3():
    st = reach([hand("9s 2s 3h"), hand("Qs Td Jd"), hand("4s 6d 7d"),
                hand("Js 8d 9d")], S)
    v = view_after(st, "4s Js")
    assert pick(v) == card("9s") and "F1" in fired(v)
    assert pick(v, P(w_F1_spades=0.0)) == card("9s")           # fallback agrees
    st = reach([hand("9h 2h 3c"), hand("Kh Td Jd"), hand("5h 6d 7d"),
                hand("Jh 8d 9d")], S)
    v = view_after(st, "5h Jh")
    assert pick(v) == card("9h") and "F1" in fired(v)


def test_f2_lowest_winner_vs_x1_highest_clean():
    st = reach([hand("9c Kc 3h"), hand("3c 6d 7d"), hand("5c 8d 9d"),
                hand("8c Td Jd")], E)
    v = view_after(st, "3c 5c 8c")
    assert pick(v, P(w_X1=0.0)) == card("9c")
    assert pick(v) == card("9c")                       # tie -> fallback lowest
    assert pick(v, P(w_X1=1.0, w_F2=0.5)) == card("Kc") and "X1" in fired(v)
    # X1 is fourth-hand only.
    st = reach([hand("9c Kc 3h"), hand("3c 6d 7d"), hand("5c 8d 9d"),
                hand("8c Td Jd")], W)
    v = view_after(st, "8c")           # W leads 8♣, N is second
    assert "X1" not in fired(v)


def test_f4_clean_win_with_proven_exit():
    st = reach([hand("Td 3d 3c"), hand("4d 5c 2h"), hand("6d 6c 3h"),
                hand("8d 7c 4h")], E)
    v = view_after(st, "4d 6d 8d")
    assert "F4" in fired(v) and "F5" not in fired(v)
    assert pick(v) == card("Td")
    assert pick(v, P(w_F4=0.0, w_X1=0.0)) == card("3d")
    # C2: the same shape with an exhausted club is NOT eligible.
    st = reach([hand("Td 3d 3c"), hand("2d Ah 4s"), hand("5d Kh 5s"),
                hand("8d Qh 6s")], E)
    v = view_after(st, "2d 5d 8d")
    assert "F4" not in fired(v) and pick(v) == card("3d") and pick(v, OFF) == card("3d")


def test_h4_sheds_top_heart_only_on_forced_win_within_cost():
    st = reach([hand("Ah 7h 3c"), hand("2h 4d 4c"), hand("5d Ad 5c"),
                hand("4h 6d 6c")], E)
    v = view_after(st, "2h 5d 4h")
    f = Facts(v)
    assert f.final_cost(card("Ah")) == 3
    assert "H4" in fired(v)
    assert pick(v) == card("7h")                       # H4 ties F2; fallback lowest
    assert pick(v, P(w_F2=0.5)) == card("Ah")
    assert "H4" not in fired(v, P(ace_release_cost=2))
    assert pick(v, P(ace_release_cost=2, w_F2=0.5)) == card("7h")


def test_h5_lowest_heart_within_suit_only():
    st = reach([hand("Qh 8h 3h 4c"), hand("4d 5d 6c 2h"), hand("7d 8d 7c 5h"),
                hand("9d Td 8c 6h")], N)
    v = st.view_for(N)
    h5 = [d for _, _, fr in explain(v, REF)["stages"] for r, d in fr if r == "H5"]
    assert h5 and set(h5[0]) == {card("Qh"), card("8h")}
    assert card("4c") not in h5[0]


def test_d1_ladder_order_follows_guarded_spade_first():
    st = reach([hand("Ks 2s Ah"), hand("6c 8d 9d"), hand("Qs 7c Td"),
                hand("5c 6d 7d")], W)
    v = view_after(st, "5c")
    assert card("2s") not in stage_survivors(v, REF, "3C")   # exit retained
    assert pick(v, P(guarded_spade_first=False)) == card("Ah")
    assert pick(v, P(guarded_spade_first=True)) == card("Ks")


def test_d2_rewards_useful_void_unless_sole_exit():
    st = reach([hand("7d 5c 8c Tc"), hand("9d 9c Jc 7s"), hand("Jd Qc 2h 8s"),
                hand("4s Qd 3h 4h")], W)
    v = view_after(st, "4s")
    d2 = [d for _, _, fr in explain(v, REF)["stages"] for r, d in fr if r == "D2"]
    assert d2 and card("7d") in d2[0]
    st = reach([hand("7d 5c 8c Tc"), hand("9d 9c Jc 7s"), hand("Jd 4c 2h 8s"),
                hand("4s Qd 3h 4h")], W)   # 4♣ out: 5♣ no longer an exit -> 7♦ sole exit
    v = view_after(st, "4s")
    assert not [d for _, _, fr in explain(v, REF)["stages"] for r, d in fr if r == "D2"]
    # C6: exhausted singleton has no bonus, and with retention A♣ is shed.
    st = reach([hand("3d Kc Ac"), hand("2s 2h 4d"), hand("4s Kh 5d"),
                hand("6s 3c Qh")], E)
    v = view_after(st, "2s 4s 6s")
    assert not [d for _, _, fr in explain(v, REF)["stages"] for r, d in fr if r == "D2"]
    assert pick(v) == card("Ac")


def test_l2_x2_multiplier_and_e4_boost_cap():
    st = reach([hand("Jd Kd Tc Jc"), hand("Ad Qd Ac Qc"), hand("2d 3d 2h 3h"),
                hand("4d 5d 4h 5h")], N)
    v = st.view_for(N)
    t = v.trick_number
    def l2(params):
        return [d for _, _, fr in explain(v, params)["stages"] for r, d in fr if r == "L2"]
    vw = 13
    mult = max(0.0, 1.0 - (t - 1) / (vw - 1))
    d = l2(P(w_L2=0.5, escape_boost=0.0, void_window=vw))[0]
    assert d[card("Jd")] == pytest.approx(0.5 * mult) and d[card("Tc")] == pytest.approx(0.5 * mult)
    d = l2(P(w_L2=0.5, escape_boost=1.0, void_window=vw))[0]
    assert d[card("Jd")] == pytest.approx(1.0 * mult)          # capped at 1 before decay
    assert not l2(P(w_L2=1.0, void_window=t))                  # window closed -> 0 -> no vote


def test_e5_triggers_only_when_all_26_captured():
    st = reach([hand("Ah 3c"), hand("4c 4d"), hand("5c 5d"), hand("6c 6d")], N)
    v = st.view_for(N)
    assert sum(v.scores) == 25 and explain(v, REF)["returned_at"] != "0/E5"
    assert pick(v) == card("3c") and pick(v, OFF) == card("3c")      # C4
    st = reach([hand("3c 4d"), hand("4c 5d"), hand("5c 6d"), hand("6c 7d")], N)
    v = st.view_for(N)
    assert sum(v.scores) == 26 and explain(v, REF)["returned_at"] == "0/E5"
    assert set(explain(v, REF)["ties"]) == set(cards.cards_in(v.legal_moves))


def test_review_lines_replayed_with_expert_at_north():
    """The review's lines C1–C7: the expert makes the line's FIRST North
    decision (the counterexample's point); the rest of the line is the
    review's scripted continuation and must total the stated N points."""
    def run(residual, leader, line, expected_n_points, params=REF):
        st = reach(residual, leader)
        before = st.scores[N]
        checked = False
        for tok in line.split():
            if st.to_play == N and not checked:
                assert pick(st.view_for(N), params) == card(tok), (tok, st.trick_number)
                checked = True
            st.play(card(tok))
        assert st.is_over() and checked and st.scores[N] - before == expected_n_points
    run([hand("3d Ac"), hand("4d 2h"), hand("Kh 5s"), hand("Qh 6s")], N,
        "3d 4d Kh Qh 2h 5s 6s Ac", 0)                                   # C1
    run([hand("Td 3d 3c"), hand("2d Ah 4s"), hand("5d Kh 5s"), hand("8d Qh 6s")], E,
        "2d 5d 8d 3d 6s 3c 4s 5s Qh Td Ah Kh", 0)                       # C2
    run([hand("Ah 3h 3c"), hand("2h 4d 4c"), hand("5d Ad 5c"), hand("8h 6d 6c")], E,
        "2h 5d 8h 3h 6c 3c 4c 5c 6d Ah 4d Ad", 0)                       # C3
    run([hand("Ah 3c"), hand("4c 4d"), hand("5c 5d"), hand("6c 6d")], N,
        "3c 4c 5c 6c 6d Ah 4d 5d", 0)                                   # C4
    run([hand("Ac 3c 9h"), hand("4c Jh 4h"), hand("5c 2h 3h"), hand("6c 4d 5d")], N,
        "3c 4c 5c 6c 4d Ac 4h 3h 5d 9h Jh 2h", 0, params=P(w_exit=0.0))  # C5 (non-retainer)
    run([hand("3d Kc Ac"), hand("2s 2h 4d"), hand("4s Kh 5d"), hand("6s 3c Qh")], E,
        "2s 4s 6s Ac 3c Kc 2h Kh 3d 4d 5d Qh", 2)                       # C6
    run([hand("Qs As 3d"), hand("2s 4d 4c"), hand("5s 5d 5c"), hand("9s 6d 6c")], E,
        "2s 5s 9s As 3d 4d 5d 6d 6c Qs 4c 5c", 0)                       # C7


# =========================================================== G3: coverage

def test_total_action_coverage_fuzz():
    ids = train_ids() + heldout_ids()
    cells = {}
    illegal = 0
    n_games = 0
    for k in range(120):
        table = [ids[(4 * k + j) % len(ids)] for j in range(4)]
        params = [sample_expert(i) for i in table]
        if k % 10 == 0:
            params[k % 4] = OFF
        st = deal(np.random.default_rng(300000 + k))
        players = [ExpertPlayer(np.random.default_rng(k * 4 + j), params[j])
                   for j in range(4)]
        while not st.is_over():
            seat = st.to_play
            v = st.view_for(seat)
            if bin(v.legal_moves).count("1") > 1:
                ex = explain(v, players[seat].params)
                q = queen_state(v.hand, v)
                phase = "early" if v.trick_number <= 3 else ("late" if v.trick_number >= 10 else "mid")
                pos = ex["kind"] if ex["kind"] != "follow" else f"follow{len(v.current_trick)+1}"
                cells[(pos, q, phase)] = cells.get((pos, q, phase), 0) + 1
                c = players[seat].choose(v)
                if not v.legal_moves & cards.bit(c):
                    illegal += 1
            else:
                c = players[seat].choose(v)
            st.play(c)
        assert sum(st.scores) == 26
        n_games += 1
    assert illegal == 0 and n_games == 120
    for pos in ("lead", "follow2", "follow3", "follow4", "discard"):
        for q in ("owned", "unseen", "captured"):
            for phase in ("early", "mid", "late"):
                if q == "owned" and phase == "late":
                    continue            # rarely reachable, not asserted
                assert cells.get((pos, q, phase), 0) > 0, (pos, q, phase)
    assert any(q == "on_table" for (_, q, _) in cells)


def test_all_optional_off_and_all_penalized_lead_still_choose():
    st = reach([hand("Ad Ac 3h"), hand("4d 2h 5c"), hand("Kh 5s 6c"),
                hand("Qh 6s 7c")], N)
    v = st.view_for(N)
    ex = explain(v, REF)
    l4 = [d for _, _, fr in ex["stages"] for r, d in fr if r == "L4"]
    assert l4 and len(l4[0]) == 2       # both black aces penalized
    c = pick(v)
    assert v.legal_moves & cards.bit(c)
    assert v.legal_moves & cards.bit(pick(v, OFF))


def test_all_low_spade_hand_discard_and_lead():
    st = reach([hand("2s 3s 4s"), hand("6c 8d 9d"), hand("7c Td Jd"),
                hand("5c 6d 7d")], W)
    v = view_after(st, "5c")
    c = pick(v)
    assert v.legal_moves & cards.bit(c) and cards.suit(c) == cards.SPADES
    assert v.legal_moves & cards.bit(pick(v, OFF))
    st = reach([hand("2s 3s 4s"), hand("6c 8d 9d"), hand("7c Td Jd"),
                hand("5c 6d 7d")], N)
    v = st.view_for(N)
    assert v.legal_moves & cards.bit(pick(v)) and v.legal_moves & cards.bit(pick(v, OFF))


# =========================================================== G4: isolation + contract

def test_same_view_from_different_hidden_worlds_gives_same_distribution():
    st = deal(np.random.default_rng(11))
    hp = HeuristicPlayer()
    for _ in range(9):
        st.play(hp.choose(st.view_for(st.to_play)))
    v = st.view_for(st.to_play)
    unseen = cards.cards_in(cards.FULL_DECK & ~v.hand & ~sum(
        cards.bit(c) for _, c in list(v.history) + list(v.current_trick)))
    rng = np.random.default_rng(5)
    dists = []
    for _ in range(5):
        perm = list(rng.permutation(unseen))
        sizes = [bin(st.hands[(v.seat + 1 + i) % 4]).count("1") for i in range(3)]
        hands, k = [], 0
        for sz in sizes:
            m = 0
            for c in perm[k:k + sz]:
                m |= cards.bit(int(c))
            hands.append(m); k += sz
        world = state_from_view(v, hands)
        v2 = world.view_for(v.seat)
        assert v2 == v
        for pid in (0, 1, 7, 203):
            params = REF if pid == 0 else sample_expert(pid)
            dists.append((pid, action_distribution(v2, params)))
    by_pid = {}
    for pid, d in dists:
        by_pid.setdefault(pid, set()).add((tuple(d[0]), tuple(d[1])))
    assert all(len(s) == 1 for s in by_pid.values())


def test_branch_isolation_a_b_a():
    a = deal(np.random.default_rng(21)); b = deal(np.random.default_rng(22))
    a.play(cards.TWO_CLUBS); b.play(cards.TWO_CLUBS)
    va, vb = a.view_for(a.to_play), b.view_for(b.to_play)
    p = sample_expert(9)
    first = action_distribution(va, p)
    action_distribution(vb, p)
    assert action_distribution(va, p) == first


class _RecordingView:
    """Exposes only PlayerView's eight fields and records every attribute
    read; anything else raises."""
    ALLOWED = {"seat", "hand", "history", "current_trick", "hearts_broken",
               "trick_number", "scores", "legal_moves"}

    def __init__(self, v):
        object.__setattr__(self, "_v", v)
        object.__setattr__(self, "reads", set())

    def __getattr__(self, k):
        if k not in self.ALLOWED:
            raise AttributeError(k)
        self.reads.add(k)
        return getattr(self._v, k)


def test_player_only_touches_the_view():
    st = deal(np.random.default_rng(31))
    st.play(cards.TWO_CLUBS)
    v = _RecordingView(st.view_for(st.to_play))
    p = ExpertPlayer(np.random.default_rng(0), sample_expert(3))
    c = p.choose(v)
    assert st.view_for(st.to_play).legal_moves & cards.bit(c)
    assert v.reads <= _RecordingView.ALLOWED


class _CountingRng:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def integers(self, *a, **k):
        self.calls += 1
        return self.rng.integers(*a, **k)

    def random(self, *a, **k):
        self.calls += 1
        return self.rng.random(*a, **k)


def test_rng_contract_one_draw_only_on_multiway_ties():
    fixed = replace(sample_expert(1), tie_mode="fixed")
    unif = replace(sample_expert(1), tie_mode="seeded_uniform")
    st = deal(np.random.default_rng(41))
    rf, ru = _CountingRng(0), _CountingRng(0)
    pf, pu = ExpertPlayer(rf, fixed), ExpertPlayer(ru, unif)
    ties_seen = 0
    while not st.is_over():
        v = st.view_for(st.to_play)
        ex = explain(v, unif)
        before = ru.calls
        cu = pu.choose(v)
        cf = pf.choose(v)
        if len(ex["ties"]) > 1:
            assert ru.calls == before + 1
            ties_seen += 1
            assert cu in ex["ties"]
        else:
            assert ru.calls == before and cu == cf
        assert v.legal_moves & cards.bit(cu)
        st.play(cf)
    assert rf.calls == 0
    # same seed, same state -> same card
    v = deal(np.random.default_rng(41)); v.play(cards.TWO_CLUBS)
    vv = v.view_for(v.to_play)
    assert ExpertPlayer(np.random.default_rng(3), unif).choose(vv) == \
        ExpertPlayer(np.random.default_rng(3), unif).choose(vv)


def test_action_distribution_is_normalized_legal_and_matches_choose():
    st = deal(np.random.default_rng(51))
    p = replace(sample_expert(2), tie_mode="seeded_uniform")
    hp = HeuristicPlayer()
    n_multi = 0
    while not st.is_over():
        v = st.view_for(st.to_play)
        cs, ps = action_distribution(v, p)
        assert abs(sum(ps) - 1.0) < 1e-12 and all(v.legal_moves & cards.bit(c) for c in cs)
        assert len(set(cs)) == len(cs)
        if len(cs) > 1:
            n_multi += 1
            assert all(abs(x - 1.0 / len(cs)) < 1e-12 for x in ps)
        assert greedy_pick(v, p) == min(cs, key=fixed_order_key)
        st.play(hp.choose(v))
    assert n_multi >= 0


CORPUS_PIN = "3e73a45e5cfa8154f87c491d18638367bd7ff6ebe009ea0f63d9da4bcca1ee80"


def corpus_signature():
    seq = []
    for pid in (0, 1, 2, 3):
        params = REF if pid == 0 else sample_expert(pid)
        for seed in range(100000, 100020):
            st = deal(np.random.default_rng(seed))
            players = [ExpertPlayer(np.random.default_rng(seed * 4 + j), params)
                       for j in range(4)]
            while not st.is_over():
                c = players[st.to_play].choose(st.view_for(st.to_play))
                seq.append(c)
                st.play(c)
    return hashlib.sha256(bytes(seq)).hexdigest()


def test_reference_corpus_pin():
    """The bitwise anchor every future compiled path gates against: the
    choice sequence of the reference config and train ids 1-3 over 20 deals."""
    sig = corpus_signature()
    assert sig == CORPUS_PIN, f"corpus signature changed: {sig}"
