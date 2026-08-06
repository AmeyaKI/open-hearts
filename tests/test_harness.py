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
