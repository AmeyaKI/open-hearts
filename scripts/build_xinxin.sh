#!/usr/bin/env bash
# Rebuild the xinxin match environment (Phase 7 Task F2 / 7C panel).
# Reproduces the environment that produced results/xinxin_tier{1,2}*.txt.
#
# Usage: scripts/build_xinxin.sh <target-dir>
#   Creates <target-dir>/xinxin-venv and <target-dir>/open_spiel.
#   Then run matches with: <target-dir>/xinxin-venv/bin/python experiments/run_xinxin_match.py ...
#
# What this does (matching the 2026-08-29 build exactly):
#   1. python3.12 venv
#   2. clone open_spiel (upstream commit 2a870da used originally; HEAD is fine
#      unless xinxin_bot.cc has diverged -- the patch will then fail loudly)
#   3. apply experiments/xinxin_knoshooting.patch (xinxin's NATIVE no-moon rule,
#      disclosed in every match header; rules config, not a strength change)
#   4. Release build with xinxin enabled (BUILD_TYPE is an ENV VAR, not a cmake flag)
#   5. install the open-hearts project into the same venv
#   6. smoke test: one full no-pass deal with 4 xinxin bots, seats must sum to 26
#
# Notes from the original build:
#   - Apple Silicon only (this whole project is single-machine; x86 breaks bitwise gates).
#   - Release vs Testing build was a measured speed NULL (0.186 vs 0.187 s/dec);
#     we keep Release anyway. threads=off is faster AND paper-faithful.
#   - open_spiel's install.sh may brew-install python@3.14 + virtualenv as a side
#     effect; harmless, reversible with brew uninstall.
#   - Expect ~15-30 min total, dominated by the C++ build.
set -euo pipefail

TARGET="${1:?usage: build_xinxin.sh <target-dir>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-python3.12}"
mkdir -p "$TARGET"

echo "== 1/6 venv =="
"$PYTHON" -m venv "$TARGET/xinxin-venv"
VENV="$TARGET/xinxin-venv"
"$VENV/bin/pip" install --upgrade pip >/dev/null

echo "== 2/6 clone open_spiel =="
if [ ! -d "$TARGET/open_spiel" ]; then
  git clone --depth 50 https://github.com/google-deepmind/open_spiel.git "$TARGET/open_spiel"
fi
cd "$TARGET/open_spiel"

echo "== 3/6 apply kNoShooting patch =="
git apply --check "$REPO/experiments/xinxin_knoshooting.patch" || {
  echo "PATCH DOES NOT APPLY -- upstream xinxin_bot.cc changed. Inspect manually." >&2
  exit 1
}
git apply "$REPO/experiments/xinxin_knoshooting.patch"

echo "== 4/6 build open_spiel with xinxin (Release) =="
./install.sh || true   # deps; may brew-install python@3.14 as a side effect
"$VENV/bin/pip" install -r requirements.txt
BUILD_TYPE=Release OPEN_SPIEL_BUILD_WITH_XINXIN=ON "$VENV/bin/pip" install .

echo "== 5/6 install open-hearts project =="
"$VENV/bin/pip" install -e "$REPO"

echo "== 6/6 smoke test =="
"$VENV/bin/python" - <<'EOF'
import time, pyspiel
from open_spiel.python.bots import uniform_random  # noqa: F401  (import sanity)
game = pyspiel.load_game("hearts(pass_cards=False,qs_breaks_hearts=False)")
bots = [pyspiel.make_xinxin_bot(game.get_parameters(), 2000, 0.4, 50, False)
        for _ in range(4)]
state = game.new_initial_state()
t0, decisions = time.time(), 0
while not state.is_terminal():
    if state.is_chance_node():
        # harness precedent: drive chance ourselves (upstream ChanceOutcomes
        # ignores pass_cards=False); action 0 at every chance node gives a
        # legal deterministic deal for a smoke test
        action = state.legal_actions()[0]
        for b in bots: b.inform_action(state, pyspiel.PlayerId.CHANCE, action)
        state.apply_action(action)
    else:
        p = state.current_player()
        a = bots[p].step(state)
        decisions += 1
        for i, b in enumerate(bots):
            if i != p: b.inform_action(state, p, a)
        state.apply_action(a)
ret = state.returns()
pts = [26 - r for r in ret]
# OpenSpiel's returns still apply ITS moon scoring (sum 78 when one seat took
# all 26); the match harness rescores from history under our rules -- for a
# smoke test both shapes prove the build works.
assert sum(pts) in (26.0, 78.0), f"unexpected scoring, got {pts}"
print(f"SMOKE OK: points {pts} (sum {sum(pts):.0f}) | {decisions} decisions "
      f"| {(time.time()-t0)/decisions:.3f} s/dec (expect ~0.17-0.26)")
EOF
echo "DONE. Run matches with: $VENV/bin/python experiments/run_xinxin_match.py --selftest"
