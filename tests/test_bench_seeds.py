"""B0: stable benchmark seat seeds (openhearts.eval.seeds) and the pre-B0 resume guards."""
import os
import subprocess
import sys

import pytest

from openhearts.eval.seeds import (HARNESS_CBENCH, HARNESS_XINXIN, SEEDS_TOKEN,
                                   bench_seat_seed)
from openhearts.players.expert_population import DOMAIN_BENCH_SEAT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_domain_is_registered_and_distinct():
    from openhearts.players import expert_population as ep
    domains = [ep.DOMAIN_POPULATION, ep.DOMAIN_DEALS, ep.DOMAIN_SEAT_ROTATION,
               ep.DOMAIN_POLICY_TIES, ep.DOMAIN_SEARCH, DOMAIN_BENCH_SEAT]
    assert len(set(domains)) == 6 and DOMAIN_BENCH_SEAT == 6


def test_pinned_values():
    # Pinned 2026-09-26 at the B0 build; a change here changes every stable-seed row.
    assert bench_seat_seed(HARNESS_CBENCH, 0, 0, 100000, 0) == 322050628
    assert bench_seat_seed(HARNESS_CBENCH, 0, 0, 100000, 1) == 1204494360
    assert bench_seat_seed(HARNESS_XINXIN, 1, 0, 100000, 0) == 3306298078
    assert bench_seat_seed(HARNESS_XINXIN, 1, 0, 100000, 3) == 3486504738


def test_32_bit_and_sensitive_to_every_argument():
    base = bench_seat_seed(HARNESS_CBENCH, 0, 2, 100007, 1)
    assert 0 <= base < 2 ** 32
    assert bench_seat_seed(HARNESS_XINXIN, 0, 2, 100007, 1) != base
    assert bench_seat_seed(HARNESS_CBENCH, 1, 2, 100007, 1) != base
    assert bench_seat_seed(HARNESS_CBENCH, 0, 3, 100007, 1) != base
    assert bench_seat_seed(HARNESS_CBENCH, 0, 2, 100008, 1) != base
    assert bench_seat_seed(HARNESS_CBENCH, 0, 2, 100007, 2) != base


def test_independent_of_pythonhashseed():
    """The whole point of B0: two processes with different hash salts agree."""
    code = ("from openhearts.eval.seeds import *; "
            "print([bench_seat_seed(HARNESS_CBENCH, 0, r, 100000 + d, s) "
            "for r in range(4) for d in range(3) for s in range(4)], "
            "hash(('ours-minority', 100000, 0)))")
    outs = []
    for salt in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=salt)
        env["PYTHONPATH"] = os.path.join(ROOT, "src")
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                                   text=True, check=True).stdout.strip())
    seeds_a, hash_a = outs[0].rsplit(" ", 1)
    seeds_b, hash_b = outs[1].rsplit(" ", 1)
    assert seeds_a == seeds_b, "stable seeds differ across hash salts"
    assert hash_a != hash_b, "sanity: the old hash() derivation IS salt-dependent"


def test_cbench_guard_refuses_pre_b0_partial(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import run_cbench as cb
    monkeypatch.setattr(cb, "RESULTS", str(tmp_path))
    _, partial = cb._partial_paths("ours-minority")
    with open(partial, "w") as f:   # a banked, pre-B0 header: config matches, no seeds token
        f.write("# cbench direction=ours-minority n_deals=1 workers=1 n_outer=50 n_inner=20 "
                "max_simulations=1000 seed_base=100000 game=x\nour@100000 4.0\nismcts@100000 7.0\n")
    with pytest.raises(AssertionError, match="BEFORE the B0 seed repair"):
        cb.run(1, 1, "ours-minority", 50, 20, 1000, 100000, "honest")


def test_xinxin_guard_refuses_pre_b0_partial(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    import run_xinxin_match as xm
    monkeypatch.setattr(xm, "RESULTS", str(tmp_path))
    partial = os.path.join(str(tmp_path), "xinxin_tier1_ours-minority_partial.txt")
    with open(partial, "w") as f:
        f.write('tier1_ours-minority@0 J{"deals": []}\n')
    monkeypatch.setattr(sys, "argv", ["run_xinxin_match.py", "--tier", "1",
                                      "--direction", "ours-minority", "--deals", "1", "--workers", "1"])
    with pytest.raises(AssertionError, match="BEFORE the B0 seed repair"):
        xm.main()
    assert SEEDS_TOKEN == "seeds=stable-v1"
