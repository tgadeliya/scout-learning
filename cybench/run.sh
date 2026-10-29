#!/usr/bin/env bash
# Run inspect_evals/cybench with settings matching Dubois et al. (2026),
# "Seven Simple Steps for Log Analysis in AI Systems", §Cybench case study:
#   - ReAct agent with bash + python tools (inspect_evals default solver)
#   - Kali Linux Docker sandbox
#   - reasoning enabled, 2.5M tokens per sample, no message/time limits
#   - binary flag scoring, 10 independent runs (epochs)
#
# Usage:
#   ./run.sh [MODEL] [extra `inspect eval` args...]
#
# Environment overrides:
#   EPOCHS=1                 paper used 10
#   CHALLENGES=avatar,urgent comma-separated subset (default: all 39)
#   TOKEN_LIMIT=2500000
#   REASONING_EFFORT=medium
#   MAX_SANDBOXES=6          concurrent challenge environments
#   LOG_DIR=./logs
#
# Examples:
#   ./run.sh openrouter/openai/gpt-5-mini --limit 1          # smoke test
#   CHALLENGES=avatar ./run.sh openrouter/anthropic/claude-sonnet-4.5
#   EPOCHS=10 ./run.sh openrouter/openai/gpt-5               # paper scale
set -euo pipefail

cd "$(dirname "$0")"

MODEL="${1:-openrouter/openai/gpt-5-mini}"
shift || true

EPOCHS="${EPOCHS:-1}"
TOKEN_LIMIT="${TOKEN_LIMIT:-2500000}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
MAX_SANDBOXES="${MAX_SANDBOXES:-6}"
export INSPECT_LOG_DIR="${LOG_DIR:-./logs}"
# inspect_evals refuses to use the (unfiltered-network) Docker sandbox without this.
export CYBENCH_ACKNOWLEDGE_RISKS=1

task_args=(-T sandbox_type=docker)
if [[ -n "${CHALLENGES:-}" ]]; then
    task_args+=(-T "eval_names=${CHALLENGES}")
fi

exec uv run inspect eval inspect_evals/cybench \
    --model "$MODEL" \
    "${task_args[@]}" \
    --epochs "$EPOCHS" \
    --token-limit "$TOKEN_LIMIT" \
    --reasoning-effort "$REASONING_EFFORT" \
    --max-sandboxes "$MAX_SANDBOXES" \
    --fail-on-error 0.1 \
    --log-format eval \
    "$@"
