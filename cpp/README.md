# open-hearts C++ backend (Track C)

A third backend behind the existing env-var socket. `OPENHEARTS_BACKEND=cpp`
opts in; unset means the numba/pure-Python paths run exactly as before and
the extension is never imported. Plan: `CPP_PORT_PLAN.md` (local).

```
cpp/
  CMakeLists.txt            build recipe (targets: core, _hearts_cpp, tests)
  include/openhearts/*.hpp  the engine core — header-only, no Python here
  src/module.cpp            the ONLY file that includes pybind11
  tests/                    C++ unit tests (plain executables, run by ctest)
  build/                    generated, gitignored
src/openhearts/engine/backend.py   the switch + one dispatcher per ported fn
src/openhearts/_hearts_cpp.*.so    the built module (gitignored)
tests/test_cpp_gate.py             bitwise gate vs the pure-Python reference
scripts/cpp_build.sh               one command: configure, build, ctest, gate
```

Build and gate:

```bash
scripts/cpp_build.sh
```

Then both JIT modes must stay green:

```bash
.venv/bin/python -m pytest -q tests/test_cpp_gate.py
OPENHEARTS_NO_JIT=1 .venv/bin/python -m pytest -q tests/test_cpp_gate.py
```

Rules (from the plan): pure Python stays the definition of correct; every
ported function gets a row in `GATES` in `tests/test_cpp_gate.py`; no timing
number is quoted until its gate passes; `-ffp-contract=off` is already set.
