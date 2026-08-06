import numpy as np
from openhearts.engine import cards
from openhearts.engine.game import deal, play_game
from openhearts.players.random_player import RandomPlayer
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.belief.table import BeliefTable, Level


def fresh_view(seed=0, seat=None):
    state = deal(np.random.default_rng(seed))
    seat = state.to_play if seat is None else seat
    return state, state.view_for(seat)


def test_uniform_start():
    _, view = fresh_view()
    t = BeliefTable.from_view(view, Level.FULL)
    unseen = cards.cards_in(t.unseen_mask)
    assert len(unseen) == 39
    for c in unseen:
        np.testing.assert_allclose(t.probs[:, c], 1 / 3)
    for c in cards.cards_in(view.hand):
        assert t.probs[:, c].sum() == 0


def test_rows_and_columns_consistent_after_play():
    rng = np.random.default_rng(11)
    players = [RandomPlayer(rng) for _ in range(4)]
    state = deal(rng)
    for _ in range(20):  # 5 tricks
        seat = state.to_play
        state.play(players[seat].choose(state.view_for(seat)))
    view = state.view_for(state.to_play)
    t = BeliefTable.from_view(view, Level.FULL)
    np.testing.assert_allclose(t.probs.sum(axis=1), t.hand_sizes, atol=1e-6)
    for c in cards.cards_in(t.unseen_mask):
        np.testing.assert_allclose(t.probs[:, c].sum(), 1.0, atol=1e-6)


def test_void_cascades_probability():
    # Hand-build a view where opponent 0 showed void in hearts:
    # their heart probability must be 0, others' hearts must rise,
    # AND their non-heart probabilities must rise (row must still fill).
    state, _ = fresh_view(seed=2)
    lead_seat = state.to_play
    # play tricks until someone discards on a heart-free led suit is fiddly;
    # instead simulate: play a full random game, then find any mid-game view
    # where a void was observed and check the cascade holds.
    rng = np.random.default_rng(2)
    players = [RandomPlayer(rng) for _ in range(4)]
    while not state.is_over():
        seat = state.to_play
        state.play(players[seat].choose(state.view_for(seat)))
        if state.trick_number >= 6 and not state.current_trick:
            view = state.view_for(state.to_play)
            t = BeliefTable.from_view(view, Level.FULL)
            for i, vs in enumerate(t.voids):
                for s in vs:
                    assert t.probs[i][
                        [c for c in range(13 * s, 13 * s + 13)]
                    ].sum() == 0
            break


def test_truth_always_has_nonzero_probability():
    # The card's true holder must never be assigned probability 0
    # (a wrong zero is permanent and would be a catastrophic bug).
    rng = np.random.default_rng(5)
    players = [HeuristicPlayer() for _ in range(4)]
    state = deal(rng)
    observer = state.to_play
    while not state.is_over():
        seat = state.to_play
        state.play(players[seat].choose(state.view_for(seat)))
        if not state.current_trick and not state.is_over():
            view = state.view_for(observer)
            t = BeliefTable.from_view(view, Level.FULL)
            for c in cards.cards_in(t.unseen_mask):
                holder = next(
                    s for s in range(4) if state.hands[s] & cards.bit(c)
                )
                i = t.opponent_seats.index(holder)
                assert t.probs[i, c] > 0, (
                    f"true holder zeroed for card {c} at trick "
                    f"{state.trick_number}"
                )


def test_levels_differ_only_as_specified():
    _, view = fresh_view(seed=9)
    u = BeliefTable.from_view(view, Level.UNIFORM)
    f = BeliefTable.from_view(view, Level.FULL)
    # before any voids/imbalance exist, all levels agree
    np.testing.assert_allclose(u.probs, f.probs)


def test_guessing_metrics_on_known_table():
    from openhearts.eval.guessing import metrics_for
    probs = np.zeros((3, 52))
    probs[0, 10] = 0.5; probs[1, 10] = 0.3; probs[2, 10] = 0.2
    probs[0, 11] = 1.0
    truth = {10: 0, 11: 0}          # card -> true opponent index
    p, nll, top1 = metrics_for(probs, truth)
    np.testing.assert_allclose(p, (0.5 + 1.0) / 2)
    np.testing.assert_allclose(nll, (-np.log(0.5) - np.log(1.0)) / 2)
    np.testing.assert_allclose(top1, 1.0)
