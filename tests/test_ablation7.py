"""7B script gates (PHASE7_PLAN.md 7B §D7 (v)-(vi)): the smoke run on its
own `_smoke` paths, the held-out wall on every line the script prints, and
the banked-partial readers."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import run_ablation7 as a7  # noqa: E402


def test_header_and_wall():
    text = a7.header(a7.ROW_NAMES, 50, 8, a7.DEAL_SEED_BASE, probe=False)
    assert "held-out ids 201-250" in text and "trios by block (ids only)" in text
    with pytest.raises(AssertionError):
        a7.assert_no_record_leak(text + " w_Q1=0.5")
    with pytest.raises(AssertionError):
        a7.assert_no_record_leak("tie_mode='fixed'")


def test_trios_are_heldout_distinct_and_deterministic():
    t1 = a7.block_trios(20)
    t2 = a7.block_trios(20)
    assert t1 == t2 and len({frozenset(t) for t in t1}) == 20
    for trio in t1:
        assert len(set(trio)) == 3 and all(201 <= i <= 250 for i in trio)
    assert a7.trio_for_deal(0, t1) == t1[0] and a7.trio_for_deal(26, t1) == t1[1]


def test_banked_readers_key_by_deal_index():
    a5 = a7.banked_ablation5_full()
    cb = a7.banked_cbench_ours()
    if a5:
        assert min(a5) == 0 and all(isinstance(v, float) for v in a5.values())
    if cb:
        assert min(cb) == 0 and max(cb) < 2000


@pytest.mark.slow
def test_smoke_runs_all_rows_on_own_paths_and_never_touches_banked():
    def snap(p):
        return (os.path.getsize(p), os.path.getmtime(p)) if os.path.exists(p) else None
    before = {p: snap(p) for p in (a7.PARTIAL, a7.REPORT, a7.BANKED_ABLATION5, a7.BANKED_CBENCH)}
    r = subprocess.run([sys.executable, os.path.join(ROOT, "experiments", "run_ablation7.py"),
                        "--smoke"], capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "ablation7 REPORT" in r.stdout and "MIRROR ALARM" in r.stdout
    a7.assert_no_record_leak(r.stdout)
    assert {p: snap(p) for p in before} == before
