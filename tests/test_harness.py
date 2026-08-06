import numpy as np
from openhearts.eval.records import (
    record_from, to_line, from_line, write_records, read_records,
)
from openhearts.engine.game import deal, play_game
from openhearts.players.heuristic import HeuristicPlayer


def test_record_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    state = deal(rng)
    initial = tuple(state.hands)
    final = play_game(state, [HeuristicPlayer() for _ in range(4)])
    rec = record_from(seed=3, initial_hands=initial, final_state=final)
    assert from_line(to_line(rec)) == rec
    assert len(rec.plays) == 52
    path = tmp_path / "games.txt"
    write_records(path, [rec, rec])
    assert read_records(path) == [rec, rec]


from openhearts.eval.stats import bootstrap_ci
from openhearts.eval.harness import rotated_match
from openhearts.players.heuristic import HeuristicPlayer


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(6.5, 2.0, size=500)
    mean, lo, hi = bootstrap_ci(data, rng=np.random.default_rng(1))
    assert lo < mean < hi
    assert abs(mean - 6.5) < 0.3


def test_harness_control_heuristic_vs_itself():
    # The broken-harness alarm: heuristic vs heuristic must average ~6.5.
    per_deal = rotated_match(
        deal_seeds=list(range(150)),
        bot_factory=lambda: HeuristicPlayer(),
        opp_factory=lambda: HeuristicPlayer(),
    )
    mean, lo, hi = bootstrap_ci(per_deal)
    assert lo <= 6.5 <= hi, f"harness control failed: CI ({lo:.2f},{hi:.2f})"
