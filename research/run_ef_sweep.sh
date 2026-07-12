#!/usr/bin/env bash
# run_ef_sweep.sh — Build, run ef sweep, plot results.
#
# ONE-HIT COMMAND:
#   EXPERIMENTS_ROOT=/path/to/experiments bash research/run_ef_sweep.sh
#
# What it does:
#   1. Builds ef_sweep via CMake.
#   2. Runs the sweep — stdout (RESULT lines) → research/ef_sweep.log,
#      stderr (progress) shown live in terminal.
#   3. Calls plot_ef_sweep.py → research/ef_sweep_3d.png + ef_sweep_2d.png

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
LOG_FILE="$REPO_ROOT/research/ef_sweep.log"
EF_SWEEP_BIN="$BUILD_DIR/ef_sweep"

if [[ -z "${EXPERIMENTS_ROOT:-}" ]]; then
    echo "[ERROR] EXPERIMENTS_ROOT is not set."
    echo "  export EXPERIMENTS_ROOT=/path/to/your/experiments"
    exit 1
fi

echo "[build] target ef_sweep"
cmake --build "$BUILD_DIR" --target ef_sweep -j"$(nproc)"

[[ -x "$EF_SWEEP_BIN" ]] || { echo "[ERROR] Build failed."; exit 1; }

ln -sf "$EF_SWEEP_BIN" "$BUILD_DIR/run"

echo "[run]   ef_sweep → $LOG_FILE  (progress on stderr)"
"$EF_SWEEP_BIN" > "$LOG_FILE"

echo "[plot]  plot_ef_sweep.py"
python3 "$REPO_ROOT/research/plot_ef_sweep.py" "$LOG_FILE"

echo "[done]  $LOG_FILE  |  research/ef_sweep_3d.png  |  research/ef_sweep_2d.png"
