"""B6: gates for the two Perilune harnesses (Tier 1 match, Tier 0 belief audit).

Everything here runs at toy sizes on offset seeds inside pytest's tmp directory -- these are
plumbing gates (scoring arithmetic, determinism, pool/resume/header paths), NOT measurements.
Skipped unless scripts/build_perilune.sh has been run.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from perilune import adapter  # noqa: E402

if not adapter.available():
    pytest.skip("Perilune checkout/weights absent (run scripts/build_perilune.sh)",
                allow_module_level=True)
pytest.importorskip("torch")

import run_perilune_belief as rb  # noqa: E402
import run_perilune_match as rm  # noqa: E402
from openhearts.eval.seeds import (HARNESS_CBENCH, HARNESS_PERILUNE, HARNESS_XINXIN,  # noqa: E402
                                   bench_seat_seed)

SMALL = (4, 2)          # toy search size: plumbing only


def test_harness_id_is_distinct_and_pinned():
    assert len({HARNESS_CBENCH, HARNESS_XINXIN, HARNESS_PERILUNE}) == 3 and HARNESS_PERILUNE == 3
    # Pinned at the B6 build (2026-10-02); a change here changes every Perilune row.
    a = bench_seat_seed(HARNESS_PERILUNE, 1, 5, 100000, 0, 0)
    b = bench_seat_seed(HARNESS_PERILUNE, 1, 5, 100000, 0, 1)
    c = bench_seat_seed(HARNESS_PERILUNE, 2, 5, 100000, 0, 0)
    assert (a, b, c) == PINNED_SEEDS
    assert len({a, b, c}) == 3


PINNED_SEEDS = (3921833763, 4007367817, 477920417)


# ------------------------------------------------------------------ Tier 1
def _sink():
    return {"ours_s": 0.0, "ours_n": 0, "perilune_s": 0.0, "perilune_n": 0,
            "fallbacks": 0, "failed_samples": 0, "inner_fallbacks": 0}


@pytest.mark.parametrize("row", list(rm.ROWS))
def test_play_deal_every_row_is_deterministic_and_counts_decisions(row):
    player = adapter.PeriluneRawPlayer("v5")
    out = []
    for _ in range(2):
        sink = _sink()
        out.append(rm.play_deal(row, "v5", 1_000_123, sink, player, SMALL))
    assert out[0] == out[1]
    mean, per_rot = out[0]
    assert len(per_rot) == 4 and all(0 <= p <= 26 for p in per_rot)
    assert mean == sum(per_rot) / 4.0
    _, subject, field = rm.ROWS[row]
    want = {"ours": 0, "perilune": 0, "heuristic": 0}
    want[subject] += 4 * 13
    want[field] += 4 * 39
    assert sink["ours_n"] == want["ours"] and sink["perilune_n"] == want["perilune"]


def test_match_rows_run_through_the_pool_resume_and_refuse_a_foreign_header(tmp_path, monkeypatch,
                                                                            capsys):
    monkeypatch.setattr(rm, "RESULTS", str(tmp_path))
    rows = ["perilune-vs-3heuristic", "heuristic-vs-3perilune", "ours-vs-3heuristic",
            "ours-vs-3perilune"]
    rm.run_rows(rows, "v5", 2, 2, 1_000_200, kind="_t", size=SMALL)
    first = {r: rm.load_row(r, "v5", 2, kind="_t")[0] for r in rows}
    assert all(sorted(v) == [0, 1] for v in first.values())
    with open(rm.partial_path(rows[0], "v5", kind="_t")) as f:
        head = f.readline()
    assert "seeds=stable-v1" in head and "harness=perilune" in head and "4x2" in head
    assert adapter.PIN_COMMIT[:12] in head

    rm.run_rows(rows, "v5", 2, 2, 1_000_200, kind="_t", size=SMALL)   # resume: nothing to redo
    assert {r: rm.load_row(r, "v5", 2, kind="_t")[0] for r in rows} == first
    with open(rm.partial_path(rows[0], "v5", kind="_t")) as f:
        assert sum(1 for ln in f if ln.startswith(rows[0] + "@")) == 1

    text = rm.report("v5", 2, kind="_t")
    assert "common yardstick" in text and "against three Perilune seats" in text
    for r in rows:
        assert r in text

    with pytest.raises(SystemExit, match="header mismatch"):        # other net, same file name
        rm.ensure_header(rm.partial_path(rows[0], "v5", kind="_t"), rows[0], "v6.1", SMALL)
    with pytest.raises(SystemExit, match="header mismatch"):        # shipped size vs toy partial
        rm.ensure_header(rm.partial_path(rows[0], "v5", kind="_t"), rows[0], "v5")


# ------------------------------------------------------------------ Tier 0
def test_score_arithmetic_on_a_hand_built_table():
    probs = np.zeros((3, 52))
    probs[:, 0] = [0.5, 0.25, 0.25]      # truth = opponent 0: hit
    probs[:, 1] = [0.0, 0.6, 0.4]        # truth = opponent 0: P = 0 (a zeroed truth)
    probs[:, 2] = [0.2, 0.005, 0.795]    # truth = opponent 1: P < 0.01, miss
    got = dict(zip(rb.COLS, rb.score(probs, {0: 0, 1: 0, 2: 1}, [0, 1, 2])))
    assert got["n"] == 3 and got["hits"] == 1 and got["zero"] == 1 and got["lt01"] == 2
    assert got["sum_p"] == pytest.approx(0.505)
    assert got["n_nll"] == 2
    assert got["sum_nll"] == pytest.approx(-np.log(0.5) - np.log(0.005))
    # cells: card 1's true cell has P=0 -> infinite loss, counted not smoothed
    assert got["inf_bce"] == 1 and got["n_bce"] == 8
    want = (-np.log(0.5) - 2 * np.log(0.75)                      # card 0
            - np.log(0.4) - np.log(0.6)                          # card 1, the two finite cells
            - np.log(0.8) - np.log(0.005) - np.log(0.205))       # card 2
    assert got["sum_bce"] == pytest.approx(want)


CARDS_SCORED_PER_GAME = sum(39 - 3 * t - i for t in range(13) for i in range(4))   # 1014


@pytest.mark.parametrize("table", ["heuristic", "perilune", "ours"])
def test_process_game_scores_every_decision_and_never_zeroes_the_truth(table):
    acc, diag = rb.process_game(table, "v5", 1_290_500, sampler_draws=4, size=SMALL)
    assert set(acc) == set(rb.CURVES) and diag["positions"] == 52
    for name, a in acc.items():
        a = np.array(a)
        assert a.shape == (13, len(rb.COLS))
        assert a[:, rb.COLS.index("n")].sum() == CARDS_SCORED_PER_GAME
        zero = a[:, rb.COLS.index("zero")].sum()
        if not name.endswith("-sampler"):            # analytic curves are truth-safe by design
            assert zero == 0, name
    uni = np.array(acc["uniform"]).sum(axis=0)
    assert uni[rb.COLS.index("sum_p")] / uni[rb.COLS.index("n")] == pytest.approx(1 / 3)
    again, _ = rb.process_game(table, "v5", 1_290_500, sampler_draws=4, size=SMALL)
    assert again == acc                               # fully replayable, samplers included


def test_belief_tables_run_through_the_pool_resume_and_report(tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "RESULTS", str(tmp_path))
    tables = ["heuristic", "perilune"]
    rb.run_tables(tables, "v5", 2, 2, 1_290_600, 4, kind="_t", size=SMALL)
    games, diag = rb.load_games("perilune", "v5", 2, kind="_t")
    assert sorted(games) == [0, 1] and diag["positions"] == 104
    rb.run_tables(tables, "v5", 2, 2, 1_290_600, 4, kind="_t", size=SMALL)      # resume
    again, _ = rb.load_games("perilune", "v5", 2, kind="_t")
    assert all(np.array_equal(games[i][c], again[i][c]) for i in games for c in games[i])
    text = rb.report("v5", 2, kind="_t")
    for needle in ("table: heuristic", "table: perilune", "perilune-deployed", "ours-sampler",
                   "curve minus exact", "by trick"):
        assert needle in text
    with pytest.raises(SystemExit, match="header mismatch"):     # different sampler budget
        rb.ensure_header(rb.partial_path("perilune", "v5", kind="_t"),
                         rb.header_line("perilune", "v5", 256, SMALL))
