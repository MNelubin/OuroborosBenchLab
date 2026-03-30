#!/usr/bin/env bash
# run_both.sh — Run PinchBench + OuroborosBench on the same model and compare
set -euo pipefail

MODEL="${1:-anthropic/claude-sonnet-4-6}"
SUITE="${2:-automated-only}"

[[ -z "${OPENROUTER_API_KEY:-}" ]] && { echo "ERROR: export OPENROUTER_API_KEY first"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Running OpenClaw (PinchBench)           ║"
echo "╚══════════════════════════════════════════╝"
cd "$WORKDIR/pinchbench"
./scripts/run.sh --model "$MODEL" --suite "$SUITE" --no-upload
OC_RESULT=$(ls -t results/*.json 2>/dev/null | head -1)

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Running Ouroboros (OuroborosBench)      ║"
echo "╚══════════════════════════════════════════╝"
cd "$SCRIPT_DIR"
python benchmark_ouroboros.py \
    --model "$MODEL" \
    --suite "$SUITE" \
    --no-upload
OB_RESULT=$(ls -t results/*.json 2>/dev/null | head -1)

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Comparison                               ║"
echo "╚══════════════════════════════════════════╝"
python compare_results.py "$OC_RESULT" "$OB_RESULT"
