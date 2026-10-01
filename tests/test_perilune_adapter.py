"""B6: Perilune adapter correctness gates (registry requirement: adapter tests before any run).

Skipped unless `scripts/build_perilune.sh` has been run (pinned checkout + their compiled engine
+ the two sha256-pinned checkpoints under <repo>/external/perilune, or $PERILUNE_DIR).

The reference implementation is THEIR code: `hearts_env` is their HeartsEnv.hpp compiled as-is,
and the raw-play reference re-enacts their `orchestrator.play_round` / `_obs_for_net` line by
line on their engine with their network. Our adapter must agree with both exactly.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from perilune import adapter  # noqa: E402

if not adapter.available():
    pytest.skip("Perilune checkout/weights absent (run scripts/build_perilune.sh)",
                allow_module_level=True)
sys.path.insert(0, os.path.join(adapter.PERILUNE_DIR, "build"))
hearts_env = pytest.importorskip("hearts_env")
torch = pytest.importorskip("torch")

from openhearts.belief.table import BeliefTable, Level  # noqa: E402
from openhearts.engine import cards  # noqa: E402
from openhearts.engine.game import deal, play_game  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402
from openhearts.players.random_player import RandomPlayer  # noqa: E402

torch.set_num_threads(1)


def _their_env_for(state):
    """Their engine, no passing (hold), holding exactly our deal."""
    env = hearts_env.HeartsEnv(seed=1, enable_passing=False)
    env.reset()
    env.set_deal([cards.cards_in(h) for h in state.hands])
    assert not env.is_passing() and env.get_pass_direction() == 3
    return env


def _their_legal(env):
    return sorted(a for a in env.get_legal_actions() if a != -1)


def _lockstep(seed, players, check):
    """Play one deal in both engines in lockstep; call check(state, env) before every play."""
    state = deal(np.random.default_rng(seed))
    env = _their_env_for(state)
    done = False
    while not state.is_over():
        assert not done
        assert env.get_current_player() == state.to_play
        check(state, env)
        card = players[state.to_play].choose(state.view_for(state.to_play))
        state.play(card)
        done = env.step(card).done
    assert done
    return state, env


def _check_obs(state, env):
    seat = state.to_play
    assert _their_legal(env) == cards.cards_in(state.view_for(seat).legal_moves)
    for s in range(4):                                   # every seat's view, not only the actor's
        view = state.view_for(s)
        theirs = np.asarray(env.observe_for(s), dtype=np.float32)
        ours = adapter.encode_obs550(view)
        assert ours.dtype == np.float32 and ours.shape == (550,)
        bad = np.flatnonzero(theirs != ours)
        assert bad.size == 0, f"obs550 mismatch seat {s} dims {bad[:10]}"
        theirs_ext = np.asarray(env.observe_ext_for(s), dtype=np.float32)
        ours_ext = adapter.encode_ext326(view)
        bad = np.flatnonzero(theirs_ext != ours_ext)
        assert bad.size == 0, f"ext326 mismatch seat {s} dims {bad[:10]}"
    # belief-label layout: their ground-truth planes are the relative opponents +1, +2, +3
    labels = np.asarray(env.observe_opponent_hands(), dtype=np.float32).reshape(3, 52)
    for k in range(3):
        assert cards.cards_in(state.hands[(seat + 1 + k) % 4]) == list(np.flatnonzero(labels[k]))


def _moon_adjusted(scores):
    return [0 if s == 26 else 26 for s in scores] if 26 in scores else list(scores)


@pytest.mark.parametrize("kind", ["random", "heuristic"])
def test_observation_and_legal_moves_match_their_engine_bit_for_bit(kind):
    n = 25 if kind == "random" else 15
    for seed in range(4100, 4100 + n):
        if kind == "random":
            rng = np.random.default_rng(seed + 7)
            players = [RandomPlayer(rng)] * 4
        else:
            players = [HeuristicPlayer()] * 4
        state, env = _lockstep(seed, players, _check_obs)
        # same trick resolution and points; their engine then applies the moon rule, ours never does
        assert list(env.get_round_scores()) == _moon_adjusted(state.scores)
        assert sum(state.scores) == 26


def test_first_trick_void_quirk_is_reproduced_and_documented():
    """Their tracker does NOT record a trick-1 discard as a club void; our exact table does."""
    for seed in range(5000, 5400):
        state = deal(np.random.default_rng(seed))
        short = [s for s in range(4) if not state.hands[s] & cards.SUIT_MASK[cards.CLUBS]]
        if not short:
            continue
        env = _their_env_for(state)
        for _ in range(4):                                # play out trick 1
            card = HeuristicPlayer().choose(state.view_for(state.to_play))
            state.play(card)
            env.step(card)
        observer = (short[0] + 1) % 4
        view = state.view_for(observer)
        assert not adapter.their_void_tracker(view)[short[0], cards.CLUBS]
        assert np.array_equal(np.asarray(env.observe_for(observer), dtype=np.float32),
                              adapter.encode_obs550(view))
        table = BeliefTable.from_view(view, Level.FULL)
        k = table.opponent_seats.index(short[0])
        assert cards.CLUBS in table.voids[k]              # we know; their tracker does not
        return
    pytest.fail("no club-void deal found in 400 seeds")


def _their_play_round(env, net, obs_dim):
    """orchestrator.play_round + _obs_for_net, re-enacted on their engine with their net."""
    plays = []
    done = False
    while not done:
        obs = np.asarray(env.observe(), dtype=np.float32)
        if obs_dim == 882:
            obs = np.concatenate([obs[:550], np.zeros(6, dtype=np.float32),
                                  np.asarray(env.observe_ext(), dtype=np.float32)])
        mask = torch.zeros((1, 52), dtype=torch.bool)
        for a in env.get_legal_actions():
            if a != -1:
                mask[0, a] = True
        with torch.no_grad():
            logits, _ = net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0), mask)
        action = int(torch.argmax(logits, dim=1).item())
        plays.append((env.get_current_player(), action))
        done = env.step(action).done
    return plays


@pytest.mark.parametrize("name,n_deals", [("v5", 6), ("v6.1", 3)])
def test_raw_player_reproduces_their_play_round_card_for_card(name, n_deals):
    net, obs_dim, _ = adapter.load_net(name)
    for seed in range(4300, 4300 + n_deals):
        state = deal(np.random.default_rng(seed))
        theirs = _their_play_round(_their_env_for(state), net, obs_dim)
        final = play_game(state, [adapter.PeriluneRawPlayer(name)] * 4)
        assert list(final.history) == theirs
        assert sum(final.scores) == 26


def test_checkpoints_and_source_are_the_pinned_bytes():
    for name, (fname, sha, _) in adapter.NETS.items():
        assert adapter.sha256_of(adapter.weights_path(name)) == sha, fname
    head = subprocess.run(
        ["git", "-C", os.path.join(adapter.PERILUNE_DIR, "hearts_simulator"), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert head == adapter.PIN_COMMIT


def _midgame_views(seed, stops=(0, 5, 14, 27, 38, 47)):
    state = deal(np.random.default_rng(seed))
    n = 0
    while not state.is_over():
        if n in stops:
            yield state, state.view_for(state.to_play)
        state.play(HeuristicPlayer().choose(state.view_for(state.to_play)))
        n += 1


def test_belief_readings_are_well_formed_and_never_zero_the_truth():
    for seed in (4400, 4401, 4402):
        for state, view in _midgame_views(seed):
            unseen = cards.cards_in(adapter.unseen_mask(view))
            r = adapter.belief_readings("v5", view)
            assert set(r) == {"raw", "normalized", "deployed"}
            for probs in r.values():
                assert probs.shape == (3, 52) and np.isfinite(probs).all()
                seen = [c for c in range(52) if c not in unseen]
                assert not probs[:, seen].any()
            assert ((r["raw"][:, unseen] > 0) & (r["raw"][:, unseen] < 1)).all()
            assert np.allclose(r["normalized"][:, unseen].sum(axis=0), 1.0)
            assert np.allclose(r["deployed"][:, unseen].sum(axis=0), 1.0)
            for c in unseen:                               # truth survives every reading
                holder = next(s for s in range(4) if state.hands[s] & cards.bit(c))
                k = (holder - view.seat - 1) % 4
                assert r["deployed"][k, c] > 0 and r["normalized"][k, c] > 0
            # The ensemble's belief IS its default (v5) component's (HeartsHybrid.forward_all) --
            # but fed the 556 prefix, i.e. WITH a zero match context. That is not the same as
            # the standalone 550 path: a 550 input skips match_proj entirely, a zero context
            # still adds the trained match_proj BIAS. Small (~1e-3 in probability) and theirs
            # by construction; pinned here so nobody "fixes" it later.
            if seed == 4400:
                v5, _, _ = adapter.load_net("v5")
                obs556 = torch.from_numpy(np.concatenate(
                    [adapter.encode_obs550(view), np.zeros(6, dtype=np.float32)])).unsqueeze(0)
                with torch.no_grad():
                    _, _, bel = v5.forward_all(obs556, torch.ones(1, 52, dtype=torch.bool))
                via_v5 = torch.sigmoid(bel[0].double()).numpy().reshape(3, 52)
                assert np.allclose(adapter.belief_sigmoid("v6.1", view), via_v5, atol=1e-9)
                gap = np.abs(adapter.belief_sigmoid("v5", view) - via_v5).max()
                assert 0.0 < gap < 0.05


def test_sampler_port_respects_hand_sizes_and_their_voids():
    rng = np.random.default_rng(11)
    for state, view in _midgame_views(4410, stops=(3, 18, 33, 45)):
        raw = adapter.belief_sigmoid("v5", view)
        caps = adapter.opponent_hand_sizes(view)
        voids = adapter.their_void_tracker(view)
        got = 0
        for _ in range(40):
            owner = adapter.sample_owner_assignment(raw, view, rng)
            if owner is None:
                continue
            got += 1
            assert sorted(owner) == cards.cards_in(adapter.unseen_mask(view))
            assert [sum(1 for k in owner.values() if k == j) for j in range(3)] == caps
            for c, k in owner.items():
                assert not voids[(view.seat + 1 + k) % 4, c // 13]
        assert got > 0
        marg, unweighted = adapter.sampler_marginals("v5", view, rng, n_draws=32)
        assert np.allclose(marg.sum(axis=1), caps) and 0.0 <= unweighted <= 1.0
        assert np.allclose(marg[:, cards.cards_in(adapter.unseen_mask(view))].sum(axis=0), 1.0)


def test_raw_player_is_deterministic_and_legal_in_our_engine():
    for name in ("v5", "v6.1"):
        runs = []
        for _ in range(2):
            state = deal(np.random.default_rng(4500))
            runs.append(tuple(play_game(state, [adapter.PeriluneRawPlayer(name)] * 4).history))
        assert runs[0] == runs[1] and len(runs[0]) == 52
