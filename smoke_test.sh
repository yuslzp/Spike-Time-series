#!/usr/bin/env bash
# Minimal smoke test for the electricity dataset OOM fixes.
# Runs 3 epochs with h=96 (worst-case output size) on GPU 0, no wandb.
#
# Usage:
#   bash smoke_test.sh [--horizon 96]  # default 96

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HORIZON=${HORIZON:-96}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --horizon) HORIZON="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

OUT="$ROOT/outputs/smoke_electricity_h${HORIZON}"
echo "=== Smoke test: electricity h=${HORIZON}, output -> $OUT ==="

PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
MPLCONFIGDIR="$ROOT/.cache/matplotlib" \
conda run -n SeqSNN env MPLCONFIGDIR="$ROOT/.cache/matplotlib" PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" WANDB_MODE=disabled python -m HybridSNN.entry.run \
    "$ROOT/exp/forecast/hybrid_snn/hybrid_snn_electricity_run.yml" \
    --runtime.use_cuda true \
    --runtime.seed 0 \
    --data.horizon "$HORIZON" \
    --runner.max_epoches 3 \
    --runner.early_stop 3 \
    --runner.batch_size 48 \
    --runner.wandb_enabled false \
    --runner.viz_every 0 \
    --runtime.output_dir "$OUT"

echo "=== Smoke test PASSED ==="
