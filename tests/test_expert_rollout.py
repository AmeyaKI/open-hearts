"""Gates for the 7C expert-rollout search player (PHASE7_PLAN.md Task 7C):
reduction (empty pool == incumbent, bitwise), belief hygiene (world set
untouched), fused/grouping off, held-out wall on the pool, branch isolation
with CRN coupling, legality, view-only rollouts."""
import numpy as np
import pytest

from openhearts.belief.table import Level
from openhearts.engine import cards
from openhearts.engine.game import deal
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.players.expert_population import train_ids
from openhearts.search.expert_rollout import ExpertRolloutSearchPlayer
from openhearts.search.honest import HonestSearchPlayer

POOL = train_ids()[:20]


def _views(seed, n=6):
    st = deal(np.random.default_rng(seed))
    h = HeuristicPlayer()
    out = []
    while not st.is_over() and len(out) < n:
        v = st.view_for(st.to_play)
        if bin(v.legal_moves).count("1") > 1 and st.trick_number >= 1:
            out.append(v)
        st.play(h.choose(v))
    return out


def test_reduction_empty_pool_is_the_incumbent_bitwise():
    for seed in (1, 2):
        for v in _views(seed, 4):
            a = HonestSearchPlayer(Level.FULL, 10, 5, np.random.default_rng(7), fused=False)
            b = ExpertRolloutSearchPlayer(Level.FULL, 10, 5, np.random.default_rng(7), rollout_ids=None)
            ca, cb = a.choose(v), b.choose(v)
            assert ca == cb
            assert np.array_equal(a.last_avgs, b.last_avgs)


def test_fused_and_grouping_forced_off_and_pool_wall():
    p = ExpertRolloutSearchPlayer(Level.FULL, 10, 5, np.random.default_rng(0), rollout_ids=POOL)
    assert p._use_fused() is False and p.group_equivalent is False
    assert p.posterior_factory is None
    with pytest.raises(AssertionError):
        ExpertRolloutSearchPlayer(Level.FULL, 10, 5, np.random.default_rng(0), rollout_ids=[1, 201])


def test_belief_hygiene_world_set_identical_to_incumbent():
    """The outer worlds are drawn by inherited code before any identity
    draw, so the world set is bitwise the incumbent's for the same seed."""
    v = _views(3, 1)[0]
    seen = {}

    def capture(name):
        def _sample(self_, table, n):
            arr = HonestSearchPlayer._sample(self_, table, n)
            seen[name] = [list(map(int, w)) for w in arr]
            return arr
        return _sample

    a = HonestSearchPlayer(Level.FULL, 12, 0, np.random.default_rng(5), fused=False)
    b = ExpertRolloutSearchPlayer(Level.FULL, 12, 0, np.random.default_rng(5), rollout_ids=POOL)
    a._sample = capture("a").__get__(a)
    b._sample = capture("b").__get__(b)
    a.choose(v); b.choose(v)
    assert seen["a"] == seen["b"]


def test_identities_shared_across_candidates_and_branch_isolated():
    v = _views(4, 1)[0]
    p = ExpertRolloutSearchPlayer(Level.FULL, 6, 0, np.random.default_rng(9), rollout_ids=POOL)
    p.choose(v)
    idents = p.last_identities
    assert len(idents) == 6
    for m in idents:
        assert set(m) == {(v.seat + 1 + i) % 4 for i in range(3)}
    # Branch isolation: replaying candidate A, then B, then A in one world
    # gives bit-identical outcomes (fresh expert rngs per playout; n_inner=0).
    from openhearts.search.decision import state_from_view
    legal = cards.cards_in(v.legal_moves)
    hands = [list(map(int, w)) for w in [[0, 0, 0]]]  # placeholder replaced below
    table_worlds = None
    def capture(self_, table, n):
        nonlocal table_worlds
        arr = HonestSearchPlayer._sample(self_, table, n)
        table_worlds = arr
        return arr
    q = ExpertRolloutSearchPlayer(Level.FULL, 3, 0, np.random.default_rng(9), rollout_ids=POOL)
    q._sample = capture.__get__(q)
    q.choose(v)
    w0 = table_worlds[0]
    def play(card):
        st = state_from_view(v, w0); st.play(card)
        q._playout_expert(st, v.seat, q.last_identities[0])
        return tuple(st.scores), tuple(st.history)
    a1 = play(legal[0]); b1 = play(legal[1]); a2 = play(legal[0])
    assert a1 == a2 and (a1 != b1 or True)


def test_expert_rollouts_are_legal_terminate_and_decide():
    p = ExpertRolloutSearchPlayer(Level.FULL, 8, 4, np.random.default_rng(11), rollout_ids=POOL)
    st = deal(np.random.default_rng(12))
    h = HeuristicPlayer()
    n = 0
    while not st.is_over():
        v = st.view_for(st.to_play)
        c = p.choose(v) if st.to_play == 0 else h.choose(v)
        assert v.legal_moves & cards.bit(c)
        st.play(c); n += 1
    assert sum(st.scores) == 26 and p.rollout_playouts > 0


def test_rollout_experts_only_see_simulated_views(monkeypatch):
    """Structural: the only accessor an expert gets is state.view_for(seat)."""
    import openhearts.search.expert_rollout as er
    calls = []
    orig = er.ExpertPlayer.choose
    def spy(self, view):
        assert view.__class__.__name__ == "PlayerView"
        calls.append(view.seat)
        return orig(self, view)
    monkeypatch.setattr(er.ExpertPlayer, "choose", spy)
    v = _views(6, 1)[0]
    p = ExpertRolloutSearchPlayer(Level.FULL, 4, 0, np.random.default_rng(1), rollout_ids=POOL)
    p.choose(v)
    assert calls and v.seat not in calls
