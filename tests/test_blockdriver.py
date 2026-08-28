"""Phase 7 Task F0 gates for `openhearts.eval.blockdriver` and `watchdog.py`.

The load-bearing test here is `test_resume_gate_bitwise_and_append_only`:
everything else in Task F0 (finer blocks, the auto-adopt rule, the watchdog
deadline) only matters if a paused-and-resumed run still reproduces the
uninterrupted run's numbers exactly, and never rewrites a line it already
banked. No JIT dependency, no real games: the worker below is a cheap
deterministic per-item value so the whole suite runs in the fast default
(`pytest -q`, no `-m slow`) sweep.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from openhearts.eval import blockdriver  # noqa: E402


# --------------------------------------------------------------- workers
# Module-level (not a closure/lambda) because ProcessPoolExecutor pickles by
# qualified name -- same constraint run_exploit_eval.worker documents.
def _det_worker(name, block_idx, item_indices):
    """Deterministic, cheap: one seeded float64 per item index. No games,
    no JIT, no filesystem -- just enough to exercise the checkpoint plumbing
    with values a test can compare bitwise."""
    vals = [float(np.random.default_rng(90_000 + i).random())
           for i in item_indices]
    return {"values": vals}


class _Collector:
    """on_block_done callback: scatters a completed block's values into a
    dict keyed by item index, so the test can assemble the full array
    regardless of the (nondeterministic) order blocks complete in."""

    def __init__(self):
        self.by_index = {}

    def __call__(self, block_idx, item_indices, payload):
        for idx, v in zip(item_indices, payload["values"]):
            self.by_index[idx] = v

    def array(self, n):
        return np.array([self.by_index[i] for i in range(n)], dtype=np.float64)


class _InterruptAfter:
    """Wraps a _Collector; raises once `at` blocks have landed, simulating a
    killed/paused run. The collector still records everything seen before
    the raise, matching what a real run's in-memory state would hold."""

    def __init__(self, collector, at):
        self.collector = collector
        self.at = at
        self.seen = 0

    def __call__(self, block_idx, item_indices, payload):
        self.collector(block_idx, item_indices, payload)
        self.seen += 1
        if self.seen >= self.at:
            raise RuntimeError("simulated interrupt for the resume gate test")


# ------------------------------------------------------- the resume gate
def test_resume_gate_bitwise_and_append_only(tmp_path):
    """Run to completion uninterrupted (dir A) vs paused-then-resumed (dir
    B); the two must agree bitwise, and B's pre-interrupt lines must survive
    the resume unchanged (append-only, never rewritten).

    NOTE (recorded design decision, per PHASE7_PLAN.md Task F0): whole-file
    BYTE equality between A's partial and B's partial is NOT the gate --
    completion order across worker processes legitimately varies, so the two
    files can bank their blocks in different orders. The gate is (1) parsed
    per-item values agree bitwise between A and B, and (2) every line B had
    banked before the interrupt is still present, unmodified, after the
    resume (checked as an exact string-for-string prefix, since append-only
    means existing lines never move).
    """
    name = "T"
    n_items = 23          # not a multiple of block_size: exercises a short
    block_size = 5         # final block, same as the real experiment scripts.
    n_blocks = -(-n_items // block_size)
    assert n_blocks == 5

    # --- dir A: uninterrupted -------------------------------------------
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    path_a = str(dir_a / "T_partial.txt")
    collector_a = _Collector()
    used_block_a = blockdriver.run_blocks(
        name, n_items, block_size, _det_worker, workers=3,
        partial_path=path_a, on_block_done=collector_a)
    assert used_block_a == block_size
    v1 = collector_a.array(n_items)
    assert len(collector_a.by_index) == n_items

    # --- dir B: interrupted after >= 2 blocks, then resumed -------------
    dir_b = tmp_path / "b"
    dir_b.mkdir()
    path_b = str(dir_b / "T_partial.txt")
    collector_b = _Collector()
    interrupter = _InterruptAfter(collector_b, at=2)
    with pytest.raises(RuntimeError, match="simulated interrupt"):
        blockdriver.run_blocks(
            name, n_items, block_size, _det_worker, workers=3,
            partial_path=path_b, on_block_done=interrupter)

    with open(path_b) as f:
        lines_after_interrupt = f.readlines()
    banked_after_interrupt = blockdriver.load_partial(path_b, name)
    # Strict subset: some blocks landed, not all of them.
    assert 0 < len(banked_after_interrupt) < n_blocks
    assert len(lines_after_interrupt) == len(banked_after_interrupt)

    # Resume: same name/items/block_size/worker, fresh collector (a real
    # resumed process would also start with empty in-memory state and rely
    # on `on_block_done` firing for the banked blocks too, which it does).
    collector_b2 = _Collector()
    used_block_b = blockdriver.run_blocks(
        name, n_items, block_size, _det_worker, workers=3,
        partial_path=path_b, on_block_done=collector_b2)
    assert used_block_b == block_size
    v2 = collector_b2.array(n_items)
    assert len(collector_b2.by_index) == n_items

    with open(path_b) as f:
        lines_after_resume = f.readlines()
    # Append-only: every line banked before the interrupt is still there,
    # unmodified, in the same position.
    assert lines_after_resume[:len(lines_after_interrupt)] == lines_after_interrupt
    assert len(lines_after_resume) == n_blocks

    # The bitwise gate itself: same values, exactly, not merely close.
    # `tobytes()` equality is the literal bit pattern, not `np.array_equal`'s
    # elementwise `==` (which would also be true here, but the raw bytes
    # comparison is the one that can't be fooled by e.g. NaN handling).
    #
    # HONESTY NOTE: `_det_worker` is a pure function of item index alone, so
    # this particular assertion would ALSO pass if the resume silently
    # ignored the partial and recomputed every block from scratch -- it
    # cannot, by itself, prove that resuming actually reused banked work.
    # The assertions that do prove that are the two above: the append-only
    # prefix check (dir B's pre-interrupt lines are byte-identical after the
    # resume) and `len(lines_after_resume) == n_blocks` combined with
    # `len(banked_after_interrupt) < n_blocks` (so at least one block in the
    # resumed file was NOT recomputed -- it was already sitting in
    # `lines_after_interrupt`). This bitwise check adds the complementary
    # claim: recomputed-or-not, every value a real run would report is
    # exactly reproducible.
    assert v1.dtype == np.float64 and v2.dtype == np.float64
    assert v1.tobytes() == v2.tobytes(), "resumed run produced different values"


# --------------------------------------------------------- auto-adopt
def test_auto_adopt_banked_block_size(tmp_path):
    """A partial banked at block_size=25 resumes and completes correctly
    when the caller's default has since changed to 5, WITHOUT passing a
    matching --block -- the whole point of the auto-adopt rule."""
    name = "T"
    path = str(tmp_path / "T_partial.txt")
    n_items = 30
    old_block_size = 25
    # Hand-write one banked block at the OLD size, as if it were left over
    # from before DEFAULT_BLOCK_SIZE changed 25 -> 5.
    seeds = list(range(old_block_size))
    payload = {"block_size": old_block_size,
               "values": [float(np.random.default_rng(90_000 + i).random())
                         for i in seeds]}
    blockdriver.append_partial(path, name, 0, payload)

    collector = _Collector()
    used = blockdriver.run_blocks(
        name, n_items, blockdriver.DEFAULT_BLOCK_SIZE, _det_worker,
        workers=2, partial_path=path, on_block_done=collector)
    assert used == old_block_size, "should have adopted the banked size"
    v = collector.array(n_items)
    expected = np.array([float(np.random.default_rng(90_000 + i).random())
                         for i in range(n_items)])
    assert np.array_equal(v, expected)


def test_explicit_block_mismatch_raises(tmp_path):
    """The same banked-25 partial, but the caller pins block_size=5 on
    purpose (explicit_block=True) -- this must be a loud failure, not a
    silent adopt, exactly like run_exploit_eval.py's behavior before F0."""
    name = "T"
    path = str(tmp_path / "T_partial.txt")
    payload = {"block_size": 25,
               "values": [1.0] * 25}
    blockdriver.append_partial(path, name, 0, payload)

    with pytest.raises(AssertionError):
        blockdriver.run_blocks(
            name, 30, 5, _det_worker, workers=1, partial_path=path,
            on_block_done=None, explicit_block=True)


# --------------------------------------------------------- watchdog unit
def test_watchdog_first_block_deadline_fires_once_and_counts():
    """No partial ever grows; once k*block_eta elapses (by the INJECTED
    clock -- no sleeping), the deadline must fire exactly once and be
    counted, not merely printed."""
    from watchdog import WatchState, check_once  # noqa: E402 (sys.path above)

    state = WatchState("does/not/matter.txt")
    counters = {"checks": 0, "first_block_firings": 0,
               "stall_firings": 0, "mem_alerts": 0}
    logged = []
    count_fn = lambda path: 0  # noqa: E731 -- never grows

    start_t = 1_000.0
    block_eta, k, stall_after = 10.0, 3.0, 30.0
    # Ticks before the deadline (30s): must not fire.
    for now in (1_005.0, 1_015.0, 1_025.0):
        check_once([state], now, start_t, k, block_eta, stall_after,
                  mem=0.0, mem_limit_gb=1e9, counters=counters,
                  log_fn=logged.append, count_fn=count_fn)
    assert counters["first_block_firings"] == 0
    assert not state.first_block_fired

    # At and after the deadline: fires exactly once, however many more
    # ticks happen.
    for now in (1_030.0, 1_045.0, 1_060.0):
        check_once([state], now, start_t, k, block_eta, stall_after,
                  mem=0.0, mem_limit_gb=1e9, counters=counters,
                  log_fn=logged.append, count_fn=count_fn)
    assert counters["first_block_firings"] == 1
    assert state.first_block_fired
    assert any("WATCHDOG DEADLINE FIRED (first block)" in line for line in logged)
    assert sum("WATCHDOG DEADLINE FIRED" in line for line in logged) == 1


def test_watchdog_stall_fires_once_and_counts_after_first_block():
    """A partial grows once (a real first block landing), then stops. The
    STALL alarm -- not the first-block deadline -- must fire once
    `stall_after` seconds pass with no further growth."""
    from watchdog import WatchState, check_once  # noqa: E402

    state = WatchState("fake.txt")
    counters = {"checks": 0, "first_block_firings": 0,
               "stall_firings": 0, "mem_alerts": 0}
    logged = []
    start_t = 0.0
    block_eta, k, stall_after = 10.0, 3.0, 30.0

    # One block lands almost immediately (well before the 30s deadline).
    check_once([state], 5.0, start_t, k, block_eta, stall_after, mem=0.0,
              mem_limit_gb=1e9, counters=counters, log_fn=logged.append,
              count_fn=lambda p: 1)
    assert state.last_growth_t == 5.0
    assert counters["first_block_firings"] == 0, (
        "a file that already produced a block must never fire the "
        "first-block deadline")

    # No further growth. Before stall_after (30s since last growth at t=5):
    for now in (10.0, 20.0, 34.0):
        check_once([state], now, start_t, k, block_eta, stall_after, mem=0.0,
                  mem_limit_gb=1e9, counters=counters, log_fn=logged.append,
                  count_fn=lambda p: 1)
    assert counters["stall_firings"] == 0

    # At/after 5 + 30 = 35s: fires exactly once, however many more ticks.
    for now in (35.0, 50.0, 90.0):
        check_once([state], now, start_t, k, block_eta, stall_after, mem=0.0,
                  mem_limit_gb=1e9, counters=counters, log_fn=logged.append,
                  count_fn=lambda p: 1)
    assert counters["stall_firings"] == 1
    assert state.stall_fired
    assert sum("WATCHDOG STALL FIRED" in line for line in logged) == 1


def test_watchdog_checks_counter_always_increments():
    """The house rule stated in the module docstring: an alarm (or its
    surrounding bookkeeping) that can pass by never running is a bug. Every
    call to check_once must count itself, alarm or not."""
    from watchdog import WatchState, check_once  # noqa: E402

    state = WatchState("fake.txt")
    counters = {"checks": 0, "first_block_firings": 0,
               "stall_firings": 0, "mem_alerts": 0}
    for i in range(7):
        check_once([state], float(i), 0.0, 3.0, 10.0, 30.0, mem=0.0,
                  mem_limit_gb=1e9, counters=counters, log_fn=lambda _l: None,
                  count_fn=lambda p: 0)
    assert counters["checks"] == 7


def test_watchdog_mem_alert_counts():
    from watchdog import WatchState, check_once  # noqa: E402

    state = WatchState("fake.txt")
    counters = {"checks": 0, "first_block_firings": 0,
               "stall_firings": 0, "mem_alerts": 0}
    logged = []
    check_once([state], 0.0, 0.0, 3.0, 10.0, 30.0, mem=200.0,
              mem_limit_gb=110.0, counters=counters, log_fn=logged.append,
              count_fn=lambda p: 0)
    assert counters["mem_alerts"] == 1
    assert any("WATCHDOG MEM ALERT" in line for line in logged)


# --------------------------------------------------- integration smoke
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("OPENHEARTS_NO_JIT"),
    reason="honest-FULL 50x20 at real games is minutes without the numba "
           "sampler; the claim this test checks (fresh run vs resumed-with-"
           "everything-already-banked run parse identically) doesn't need "
           "the slow mode to also be exercised here -- test_exploiter.py's "
           "JIT-mode gates already cover honest-FULL's own correctness "
           "under NO_JIT.")
def test_run_exploit_eval_smoke_block5_resume_is_a_noop(tmp_path):
    """`run_exploit_eval.py --smoke --block 5` twice in a row: the second
    invocation must report every block already banked and recompute
    nothing, and the two runs' per-deal values must parse identically."""
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    script = os.path.join(repo_root, "experiments", "run_exploit_eval.py")
    env = dict(os.environ)
    cmd = [sys.executable, script, "--smoke", "--rows", "R0,R1",
          "--deals", "3", "--workers", "2", "--block", "5"]

    def _run():
        return subprocess.run(cmd, cwd=repo_root, env=env,
                              capture_output=True, text=True, timeout=1200)

    try:
        first = _run()
        assert first.returncode == 0, first.stdout + first.stderr
        second = _run()
        assert second.returncode == 0, second.stdout + second.stderr
    finally:
        # Smoke partials are scratch files by contract (module docstring:
        # "--smoke ... never touches a real partial") -- clean up so this
        # test is hermetic and never pollutes results/.
        for row in ("R0", "R1"):
            for path in (
                os.path.join(repo_root, "results",
                             f"exploit_{row}_smoke_partial.txt"),
            ):
                if os.path.exists(path):
                    os.remove(path)
        smoke_out = os.path.join(repo_root, "results", "exploit_eval_smoke.txt")
        if os.path.exists(smoke_out):
            os.remove(smoke_out)

    assert "[resume]" in second.stdout, (
        "second run should have reported banked blocks, not recomputed them"
    )
    # Every block already banked -> the second run submits zero jobs, so the
    # per-block progress line ("[R.. n/n] block ...", printed once per
    # completed job) must never appear at all.
    assert "] block " not in second.stdout, (
        "second run recomputed at least one block instead of reusing the "
        "banked partial")
