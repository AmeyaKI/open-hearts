import numpy as np
from openhearts.engine import cards
from openhearts.engine.game import deal, play_game
from openhearts.belief.table import Level
from openhearts.search.decision import SearchPlayer, state_from_view
from openhearts.players.heuristic import HeuristicPlayer


def test_state_from_view_accounts_for_all_52_cards():
    rng = np.random.default_rng(1)
    state = deal(rng)
    players = [HeuristicPlayer() for _ in range(4)]
    for _ in range(10):
        seat = state.to_play
        state.play(players[seat].choose(state.view_for(seat)))
    view = state.view_for(state.to_play)
    sampled = [state.hands[s] for s in range(4) if s != view.seat]
    st = state_from_view(view, sampled)
    total = sum(bin(h).count("1") for h in st.hands)
    total += len(st.history) + len(st.current_trick)
    assert total == 52
    assert st.to_play == view.seat


def test_search_player_completes_a_game_legally():
    rng = np.random.default_rng(6)
    bot = SearchPlayer(Level.FULL, n_samples=10, rng=rng)
    others = [HeuristicPlayer() for _ in range(3)]
    state = play_game(deal(rng), [bot] + others)
    assert sum(state.scores) == 26


def test_single_legal_move_skips_search():
    # With one legal move the bot must answer instantly without sampling.
    rng = np.random.default_rng(8)
    bot = SearchPlayer(Level.FULL, n_samples=10_000_000, rng=rng)  # absurd N
    state = deal(np.random.default_rng(8))
    leader = state.to_play
    view = state.view_for(leader)   # trick 0: only the 2c is legal
    assert bot.choose(view) == cards.TWO_CLUBS
