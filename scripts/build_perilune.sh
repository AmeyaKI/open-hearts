#!/usr/bin/env bash
# Build the Perilune (JAkoliver/hearts_simulator, MIT) side of the B6 cross-benchmark.
#
#   scripts/build_perilune.sh [dest]        default dest: <repo>/external/perilune (gitignored)
#
# What it does, in order (idempotent; rerun any time):
#   1. clones the repository and checks out the PINNED commit (detached);
#   2. compiles their header-only C++ engine's pybind module `hearts_env` (bindings.cpp +
#      HeartsEnv.hpp, no torch/CUDA dependency) against OUR venv's Python -- used ONLY by the
#      adapter-correctness tests as the reference implementation of their observation;
#   3. downloads the two released checkpoints from their `models-v1` GitHub release and
#      verifies sha256 against the digests pinned below (which were cross-checked against the
#      release's own SHA256SUMS / ENSEMBLE_V6_1_SHA256SUMS manifests on 2026-10-02).
#
# Nothing here runs a match. The adapter (experiments/perilune/adapter.py) finds the result
# through $PERILUNE_DIR, defaulting to <repo>/external/perilune.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/external/perilune}"
PY="${PYTHON:-$ROOT/.venv/bin/python}"

REPO_URL="https://github.com/JAkoliver/hearts_simulator.git"
PIN="c395d2a77c139eb9bfa0e548c7a20b85d208c547"          # 2026-09-20, v6.1 promotion head
RELEASE_URL="https://github.com/JAkoliver/hearts_simulator/releases/download/models-v1"

# asset name -> sha256 (GitHub release digests, models-v1, published 2026-08-12;
# the 710c2102 ensemble was added to the same release on its 2026-09-20 promotion)
ASSETS=(
  "hearts_model_final.pth:7ce504ce78a728e282437041b7094c3a4eb106b06bb5becf5447e881bf445a5a"
  "hearts_ensemble_710c2102.pth:75d87a2a51f3e47ae5250480fecaf805bfa704d67466840aba2f34c009c7f6a3"
)

mkdir -p "$DEST/weights" "$DEST/build"

# 1. source at the pinned commit
if [ ! -d "$DEST/hearts_simulator/.git" ]; then
  git clone -q "$REPO_URL" "$DEST/hearts_simulator"
fi
git -C "$DEST/hearts_simulator" fetch -q origin
git -C "$DEST/hearts_simulator" checkout -q --detach "$PIN"
echo "source: $(git -C "$DEST/hearts_simulator" rev-parse HEAD)"
test "$(git -C "$DEST/hearts_simulator" rev-parse HEAD)" = "$PIN"
grep -q "^MIT License" "$DEST/hearts_simulator/LICENSE"   # license gate (registry row)

# 2. their engine as a Python module (reference observation for the adapter tests)
SUFFIX="$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
INCLUDES="$("$PY" -m pybind11 --includes)"
OUT="$DEST/build/hearts_env$SUFFIX"
if [ ! -f "$OUT" ] || [ "$DEST/hearts_simulator/HeartsEnv.hpp" -nt "$OUT" ]; then
  # shellcheck disable=SC2086
  c++ -O2 -std=c++17 -shared -fPIC -undefined dynamic_lookup $INCLUDES \
      -I"$DEST/hearts_simulator" "$DEST/hearts_simulator/bindings.cpp" -o "$OUT"
fi
PYTHONPATH="$DEST/build" "$PY" -c 'import hearts_env; e = hearts_env.HeartsEnv(seed=1, enable_passing=False); e.reset(); assert e.get_pass_direction() == 3 and not e.is_passing(); print("hearts_env: ok (hold direction, no passing phase)")'

# 3. released checkpoints, sha256-verified
for entry in "${ASSETS[@]}"; do
  name="${entry%%:*}"; want="${entry##*:}"
  path="$DEST/weights/$name"
  if [ ! -f "$path" ]; then
    echo "downloading $name ..."
    curl -sSL --fail -o "$path.part" "$RELEASE_URL/$name"
    mv "$path.part" "$path"
  fi
  got="$(shasum -a 256 "$path" | awk '{print $1}')"
  if [ "$got" != "$want" ]; then
    echo "sha256 MISMATCH for $name: got $got want $want" >&2
    exit 1
  fi
  echo "weights: $name sha256 ok (md5 $(md5 -q "$path" | cut -c1-8))"
done

echo "perilune build complete: $DEST"
