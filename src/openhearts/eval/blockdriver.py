"""The generic block-checkpointed parallel driver, extracted for Phase 7.

WHY THIS EXISTS, PLAINLY (Task F0, PHASE7_PLAN.md, Session Lessons 1+3 from
PHASE6_PLAN.md). `run_exploit_eval.py` invented a good pattern under
deadline: split a long run into fixed-size BLOCKS of items (deals), run each
block in a worker process, and append one JSON line per finished block to a
plain-text partial file so a resumed run can skip blocks it already paid for.
That pattern worked, but at BLOCK_SIZE=25 a pause could lose up to 40 minutes
of already-computed work, and the last block of an N-worker run could pin one
core alone for 40 minutes while the other N-1 sat idle waiting for it (a
straggler tail). Both are pure granularity problems: shrink the block and both
shrink with it, at the cost of more (cheap) bookkeeping lines. This module is
that fix, pulled out of `run_exploit_eval.py` so every NEW Phase 7 experiment
script gets it by import instead of by copy-paste. DEFAULT_BLOCK_SIZE = 5 is
the F0 decision: five deals lost to a pause, not twenty-five.

FILE FORMAT (unchanged from run_exploit_eval.py / run_entropy_curve.py, so
existing partials stay readable): one line per finished block,
    <name>@<block_idx> J<json>
where the JSON payload is caller-defined except for one required key,
`block_size`, which lets a resumed run detect whether the blocks on disk were
cut to the same size as the one it is about to request (see `run_blocks`).

RESUME CONTRACT. Banked blocks are read once at the start, never rewritten;
new blocks are only ever appended. This is what makes a paused-and-resumed
run reproduce the same numbers as an uninterrupted one: nothing already on
disk is ever recomputed or overwritten, so floating point results for a
banked block are bit-for-bit whatever they were when the block first
completed, regardless of how many times the run has been paused since.
"""
import concurrent.futures as cf
import json
import os
import subprocess
import time

#: The F0 decision: five items per block by default, down from the 25 that
#: run_exploit_eval.py used before this task. Callers that need a different
#: default (or that must stay bitwise-identical to an older banked layout)
#: pass their own `block_size` to `run_blocks`.
DEFAULT_BLOCK_SIZE = 5


def load_partial(path, name):
    """Read a partial file and return {block_idx: payload_dict} for `name`.

    Lines for other names (a shared partial file, or a stale name) and blank
    / comment lines are skipped. Missing file -> empty dict, not an error:
    a fresh run has no partial yet.
    """
    banked = {}
    if not os.path.exists(path):
        return banked
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            head, rest = line.split(" ", 1)
            n, b = head.rsplit("@", 1)
            if n != name or not rest.startswith("J{"):
                continue
            banked[int(b)] = json.loads(rest[1:])
    return banked


def append_partial(path, name, block_idx, payload_dict):
    """Append one completed block's line. Never rewrites, never truncates.

    `payload_dict` must already carry the keys the caller wants back on
    resume (including `block_size` -- `run_blocks` sets it before calling
    this). JSON round-trips Python float64 exactly, so no custom float
    encoding is needed for the bitwise-resume guarantee to hold.
    """
    with open(path, "a") as f:
        f.write(f"{name}@{block_idx} J{json.dumps(payload_dict)}\n")
        f.flush()


def _total_rss_gb():
    """Sum RSS (GB) over this process group -- the pattern run_exploit_eval.py
    uses (`ps -o rss= -g <pgid>`), copied verbatim so both call sites read
    the same number the same way."""
    out = subprocess.run(["ps", "-o", "rss=", "-g", str(os.getpgrp())],
                         capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split()) / (1024 ** 2)


def _call_worker(worker_fn, name, block_idx, item_indices):
    """Runs in the worker process. Tags the result with its own block_idx
    since `worker_fn`'s contract (name, block_idx, item_indices) -> payload
    doesn't otherwise let `as_completed` tell futures apart."""
    return block_idx, worker_fn(name, block_idx, item_indices)


def run_blocks(name, n_items, block_size, worker_fn, workers, partial_path,
               on_block_done=None, mem_limit_gb=100.0, explicit_block=False):
    """Generic version of run_exploit_eval.run_row.

    Splits `n_items` into blocks of `block_size`, loads whatever is already
    banked in `partial_path` under `name`, submits every non-banked block to
    a `ProcessPoolExecutor(max_workers=workers)`, and appends one partial
    line per completion. `worker_fn(name, block_idx, item_indices)` does the
    actual work and must return a JSON-serializable payload dict; it runs in
    the worker process, so it (and anything it closes over) must be
    picklable -- a module-level function, same as run_exploit_eval.worker.

    `on_block_done(block_idx, item_indices, payload)`, if given, is called
    once per block that becomes available -- first for every already-banked
    block (in the order `load_partial` returns them), then for each newly
    completed block as it finishes. This is the merge point: the caller
    reconstructs its own per-item arrays / diagnostics from `payload` there.
    If it raises, the exception propagates out of `run_blocks` immediately;
    blocks already appended stay appended (append-only), and blocks still
    running in the pool are abandoned by the exiting `with` block without
    being processed or written -- so the partial file after an interrupt
    holds a strict subset of blocks, never a partial or corrupted one.

    BLOCK-SIZE RESUME RULE (the F0 decision). If banked blocks exist and
    carry a `block_size` different from the one requested here:
      - if `explicit_block` is False (the caller didn't pin a size), the
        banked size is ADOPTED -- loudly, with a printed notice -- so old
        25-deal partials stay resumable under the new default of 5.
      - if `explicit_block` is True (the caller pinned a size on purpose),
        this raises AssertionError instead, exactly like run_exploit_eval.py
        did before this task: a deliberate mismatch is a bug, not a resume.

    Returns the block size actually used (post-adopt), since callers report
    it and it may differ from what they passed in.
    """
    banked = load_partial(partial_path, name)
    if banked:
        banked_sizes = {int(p["block_size"]) for p in banked.values()}
        assert len(banked_sizes) == 1, (
            f"{partial_path} has blocks banked under mixed block_size values "
            f"{sorted(banked_sizes)} for {name!r} -- the partial is corrupt")
        banked_size = next(iter(banked_sizes))
        if banked_size != block_size:
            if explicit_block:
                raise AssertionError(
                    f"{partial_path} was written with block_size={banked_size} "
                    f"but this run explicitly requested block_size={block_size}; "
                    f"delete the partial or rerun with block_size={banked_size}")
            print(f"[resume] adopting banked block_size={banked_size} from "
                  f"{partial_path}", flush=True)
            block_size = banked_size

    n_blocks = max(1, (n_items + block_size - 1) // block_size)

    def _items_for(b):
        return list(range(b * block_size, min(n_items, (b + 1) * block_size)))

    for b, payload in banked.items():
        if b >= n_blocks:
            continue
        if on_block_done is not None:
            on_block_done(b, _items_for(b), payload)
    if banked:
        print(f"[resume] {name}: {len(banked)} banked block(s) from "
              f"{partial_path}", flush=True)

    todo = [b for b in range(n_blocks) if b not in banked]
    t0, done = time.time(), 0
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(_call_worker, worker_fn, name, b, _items_for(b)): b
               for b in todo}
        for fut in cf.as_completed(jobs):
            b, payload = fut.result()
            payload = dict(payload)
            payload["block_size"] = block_size
            append_partial(partial_path, name, b, payload)
            done += 1
            mem = _total_rss_gb()
            print(f"[{name} {done}/{len(jobs)}] block {b} | mem={mem:.1f}GB | "
                  f"{time.time() - t0:.0f}s", flush=True)
            if mem > mem_limit_gb:
                for j in jobs:
                    j.cancel()
                raise MemoryError(f"memory {mem:.1f}GB over {mem_limit_gb}GB")
            if on_block_done is not None:
                on_block_done(b, _items_for(b), payload)
    return block_size
