import numpy as np
from openhearts.engine import cards
from openhearts.engine.game import deal
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.belief.table import BeliefTable, Level
from openhearts.sampler.sampler import sample_arrangement


def test_sample_is_legal_and_complete():
    rng = np.random.default_rng(0)
    state = deal(rng)
    view = state.view_for(state.to_play)
    t = BeliefTable.from_view(view, Level.FULL)
    result = sample_arrangement(t, rng)
    assert result is not None
    hands, attempts = result
    union = 0
    for i, h in enumerate(hands):
        assert bin(h).count("1") == t.hand_sizes[i]
        for s in t.voids[i]:
            assert h & cards.SUIT_MASK[s] == 0
        assert union & h == 0
        union |= h
    assert union == t.unseen_mask


def test_sampling_respects_midgame_constraints():
    rng = np.random.default_rng(4)
    state = deal(rng)
    players = [HeuristicPlayer() for _ in range(4)]
    for _ in range(32):  # 8 tricks in
        seat = state.to_play
        state.play(players[seat].choose(state.view_for(seat)))
    view = state.view_for(state.to_play)
    t = BeliefTable.from_view(view, Level.FULL)
    for _ in range(50):
        result = sample_arrangement(t, rng)
        assert result is not None, "sampler failed mid-game"
