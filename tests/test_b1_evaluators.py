"""B1: HeuristicRolloutEvaluator gates. pyspiel-only -- skipped in the project
venv, run in the pyspiel venv: <venv>/bin/python -m pytest tests/test_b1_evaluators.py"""
import os
import sys

import numpy as np
import pytest

pyspiel = pytest.importorskip("pyspiel")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from cbench import adapter  # noqa: E402
from cbench.evaluators import HeuristicRolloutEvaluator, mirror_from_history  # noqa: E402
from openhearts.engine.state import GameState  # noqa: E402
from openhearts.players.heuristic import HeuristicPlayer  # noqa: E402


def _random_midgame(seed, n_plays):
    game = pyspiel.load_game(adapter.GAME_STRING)
    st = game.new_initial_state()
    mirror = adapter.force_deal(st, seed=seed)
    rng = np.random.RandomState(seed)
    for _ in range(n_plays):
        if st.is_terminal():
            break
        a = rng.choice(st.legal_actions())
        adapter.apply_both(st, mirror, adapter.os_to_ours(a))
    return st, mirror


def _same(a: GameState, b: GameState):
    return (a.hands == b.hands and a.history == b.history and a.current_trick == b.current_trick
            and a.hearts_broken == b.hearts_broken and a.trick_number == b.trick_number
            and a.scores == b.scores and a.to_play == b.to_play)


def test_mirror_rebuild_equals_lockstep_mirror():
    for seed, n in [(1, 0), (2, 7), (3, 23), (4, 40), (5, 52)]:
        st, mirror = _random_midgame(800000 + seed, n)
        m, n_ill = mirror_from_history(st, strict=True)
        assert _same(m, mirror) and n_ill == 0, (seed, n)


def test_mirror_rebuild_on_resampled_world():
    st, _ = _random_midgame(800010, 15)
    sampler = pyspiel.UniformProbabilitySampler(7, 0., 1.)
    rs = st.resample_from_infostate(st.current_player(), sampler)
    m, _ = mirror_from_history(rs)
    assert sum(bin(h).count("1") for h in m.hands) == 52 - 15
    assert m.to_play == rs.current_player()


def test_evaluate_matches_python_heuristic_playout_and_terminal_rescore():
    ev = HeuristicRolloutEvaluator()
    h = HeuristicPlayer()
    for seed, n in [(11, 5), (12, 18), (13, 33), (14, 47)]:
        st, mirror = _random_midgame(800100 + seed, n)
        ref = mirror.copy()
        while not ref.is_over():
            ref.play(h.choose(ref.view_for(ref.to_play)))
        got = ev.evaluate(st)
        assert np.array_equal(got, np.array([26.0 - s for s in ref.scores])), (seed, n)
        assert abs(sum(26.0 - got)) - 26.0 < 1e-9
    st, mirror = _random_midgame(800200, 52)
    assert st.is_terminal()
    assert np.array_equal(ev.evaluate(st), np.array([26.0 - s for s in adapter.rescore(mirror)]))


def test_evaluate_is_deterministic():
    ev = HeuristicRolloutEvaluator()
    st, _ = _random_midgame(800300, 10)
    assert np.array_equal(ev.evaluate(st), ev.evaluate(st))


def test_prior_masses():
    st, mirror = _random_midgame(800400, 9)
    legal = list(st.legal_actions())
    uni = dict(HeuristicRolloutEvaluator().prior(st))
    assert set(uni) == set(legal) and abs(sum(uni.values()) - 1.0) < 1e-12
    assert all(abs(p - 1.0 / len(legal)) < 1e-12 for p in uni.values())
    eps = 0.25
    pri = dict(HeuristicRolloutEvaluator(prior_eps=eps).prior(st))
    assert set(pri) == set(legal) and abs(sum(pri.values()) - 1.0) < 1e-12
    a_h = adapter.ours_to_os(HeuristicPlayer().choose(mirror.view_for(mirror.to_play)))
    assert abs(pri[a_h] - (1 - eps + eps / len(legal))) < 1e-12
    for a in legal:
        if a != a_h:
            assert abs(pri[a] - eps / len(legal)) < 1e-12


def test_stock_rung_reproduces_stable_seed_row_on_3_deals(tmp_path, monkeypatch):
    """Regression: at the defaults the new rung knobs change nothing -- the
    first 3 deals of a fresh run equal the stable-seed replication row byte
    for byte (per-deal our@/ismcts@ lines)."""
    import run_cbench as cb
    ref = {}
    with open(os.path.join(ROOT, "results", "cbench_ours-minority_stableseed_partial.txt")) as f:
        for line in f:
            if line.startswith(("our@", "ismcts@")):
                k, v = line.split(); ref[k] = v
    monkeypatch.setattr(cb, "RESULTS", str(tmp_path))
    monkeypatch.delenv("CBENCH_ISMCTS_CFG", raising=False)
    assert cb.ismcts_cfg_tag() == ""
    cb.run(3, 1, "ours-minority", 50, 20, 1000, 100000, "honest")
    got = {}
    with open(os.path.join(str(tmp_path), "cbench_ours-minority_partial.txt")) as f:
        for line in f:
            if line.startswith(("our@", "ismcts@")):
                k, v = line.split(); got[k] = v
    assert len(got) == 6
    for k, v in got.items():
        assert ref[k] == v, (k, v, ref[k])


def test_non_default_rung_header_and_guard(tmp_path, monkeypatch):
    import json
    import run_cbench as cb
    monkeypatch.setattr(cb, "RESULTS", str(tmp_path))
    monkeypatch.setenv("CBENCH_ISMCTS_CFG", json.dumps(dict(cb.ISMCTS_DEFAULTS, rung="heur-rollout")))
    assert cb.ismcts_cfg_tag().startswith(" ismcts=heur-rollout:")
    # a stable-seed STOCK partial must not be resumed by a heur-rollout run
    _, partial = cb._partial_paths("ours-minority")
    with open(partial, "w") as f:
        f.write("# cbench direction=ours-minority n_deals=1 workers=1 n_outer=50 n_inner=20 "
                "max_simulations=1000 seed_base=100000 game=x seeds=stable-v1\n")
    with pytest.raises(AssertionError, match="different ISMCTS rung"):
        cb.run(1, 1, "ours-minority", 50, 20, 1000, 100000, "honest")


def test_inconsistent_resampled_worlds_are_evaluated_and_counted():
    """OpenSpiel's resampler ignores follow-suit/point-card constraints implied
    by the public history; the evaluator must not crash on such worlds and must
    count them. Search many resamples of a mid-game state until one is found."""
    ev = HeuristicRolloutEvaluator()
    st, _ = _random_midgame(800500, 20)
    found = 0
    for k in range(200):
        rs = st.resample_from_infostate(st.current_player(), pyspiel.UniformProbabilitySampler(k, 0., 1.))
        _, n_ill = mirror_from_history(rs)
        ret = ev.evaluate(rs)
        assert abs(sum(26.0 - ret) - 26.0) < 1e-9
        found += n_ill > 0
    assert ev.n_evaluate == 200 and ev.n_inconsistent_worlds == found
    # (found may be 0: OpenSpiel's resampler respected the history in every one of
    # 200 resamples of this state on 2026-09-27; the counter is the report-only
    # diagnostic that measures how often the tree hands us an illegal past.)


def test_unchecked_replay_tolerates_and_flags_an_illegal_play():
    from cbench.evaluators import _play_unchecked
    from openhearts.engine import cards
    from openhearts.engine.game import deal
    m = deal(np.random.RandomState(5))
    leader = m.to_play
    hand = m.hands[leader]
    bad = next(c for c in cards.cards_in(hand) if c != cards.TWO_CLUBS)   # must lead 2C
    assert _play_unchecked(m, bad) is False
    assert not (m.hands[leader] & cards.bit(bad)) and m.current_trick == [(leader, bad)]
    assert m.to_play == (leader + 1) % 4
