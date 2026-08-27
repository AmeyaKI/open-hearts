# ci-probe findings (2026-08-26) — offshore compute ruled out

Purpose: test whether GitHub Actions could validly run open-hearts benchmarks
off the owner's machine. Verdict: **NO — cross-architecture floating-point
drift breaks the project's bitwise-reproducibility standard.**

Evidence (runs on this branch):
- Run 33025568739 / 33026436091: an x86 (AMD EPYC) runner FAILED
  tests' bitwise tripwire — one belief-table probability differed from the
  arm64 reference by exactly 1 ULP (last mantissa bit), i.e. numpy's SIMD
  summation path differs by architecture. One ULP in a belief occasionally
  flips a card choice, which changes game trajectories — a benchmark there
  is a different experiment than the pre-registered one.
- Worse: one runner PASSED the same suite and another FAILED it — results
  are not even stable between GitHub's own machines (different CPU
  generations).
- The suite itself is otherwise portable: 250+ tests green on Linux/x86.
  The engine is correct cross-platform; only bitwise EQUALITY to the arm64
  bank fails.

Consequence (recorded in HANDOFF/README limitations): all benchmarks run on
the owner's M5 Max only. This branch is kept as evidence; do not merge.
