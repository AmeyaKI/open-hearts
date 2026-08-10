import numpy as np
from openhearts.belief.table import Level
from openhearts.engine import cards
from openhearts.engine.game import deal, play_game
from openhearts.players.heuristic import HeuristicPlayer
from openhearts.search.honest import HonestSearchPlayer


def test_completes_games_legally():
    rng = np.random.default_rng(3)
    bot = HonestSearchPlayer(Level.FULL, n_outer=8, n_inner=5, rng=rng)
    state = play_game(deal(rng), [bot] + [HeuristicPlayer() for _ in range(3)])
    assert sum(state.scores) == 26


def test_single_legal_move_instant():
    rng = np.random.default_rng(4)
    bot = HonestSearchPlayer(Level.FULL, n_outer=10**7, n_inner=10**7, rng=rng)
    state = deal(np.random.default_rng(8))
    view = state.view_for(state.to_play)
    assert bot.choose(view) == cards.TWO_CLUBS


def test_reduces_to_phase1_when_inner_disabled():
    # n_inner=0 must mean "no re-determinization": bitwise-identical choice
    # to Phase-1 SearchPlayer given the same rng stream.
    from openhearts.search.decision import SearchPlayer
    state = deal(np.random.default_rng(9))
    for _ in range(8):
        seat = state.to_play
        state.play(HeuristicPlayer().choose(state.view_for(seat)))
    view = state.view_for(state.to_play)
    # jit_sampler=False: "identical rng stream to Phase 1" is inherently a
    # Python-sampler property (the batch sampler seeds numba from one draw).
    a = HonestSearchPlayer(Level.FULL, n_outer=30, n_inner=0,
                           rng=np.random.default_rng(7),
                           jit_sampler=False).choose(view)
    b = SearchPlayer(Level.FULL, 30, np.random.default_rng(7)).choose(view)
    assert a == b
