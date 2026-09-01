#!/bin/bash
# One-command build + gate for the C++ backend (Track C).
#
#   scripts/cpp_build.sh            # configure, build, C++ unit tests, pytest gate
#   scripts/cpp_build.sh --no-test  # just build
#   CPP_BUILD_TYPE=Debug scripts/cpp_build.sh   # -O0 -g, for a debugger
#
# Re-running is cheap: CMake only reconfigures when CMakeLists.txt changed, and
# ninja only recompiles the .cpp files whose sources (or included headers)
# changed. Editing a header recompiles everything that includes it — this is
# why header hygiene matters in big projects.
#
# WHY NINJA, NOT MAKE (lesson from session 1): macOS ships GNU make 3.81
# (2006), which compares file times at ONE-SECOND resolution. Edit a header
# within the same second an object file was produced and make silently
# declares the object up to date — you test a stale .so and chase a bug that
# is not there. Ninja uses nanosecond mtimes and is faster besides. If you
# ever suspect staleness anyway: `rm -rf cpp/build` and rebuild.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
BUILD="$ROOT/cpp/build"

command -v cmake >/dev/null || { echo "cmake missing: brew install cmake"; exit 1; }
command -v ninja >/dev/null || { echo "ninja missing: brew install ninja"; exit 1; }
[ -x "$PY" ] || { echo "venv python missing at $PY"; exit 1; }
"$PY" -c "import pybind11" 2>/dev/null || { echo "pybind11 missing: $PY -m pip install pybind11"; exit 1; }

# configure (idempotent) — Python_EXECUTABLE pins the module to the venv's
# interpreter so the ABI tag in the .so name matches what will import it.
cmake -S "$ROOT/cpp" -B "$BUILD" -G Ninja \
      -DPython_EXECUTABLE="$PY" \
      -DCMAKE_BUILD_TYPE="${CPP_BUILD_TYPE:-Release}"

# build everything (module + test executables) in parallel
cmake --build "$BUILD" --parallel

if [ "${1:-}" = "--no-test" ]; then
    exit 0
fi

echo "== C++ unit tests (ctest) =="
ctest --test-dir "$BUILD" --output-on-failure

echo "== pytest bitwise gate =="
cd "$ROOT"
"$PY" -m pytest -q tests/test_cpp_gate.py
