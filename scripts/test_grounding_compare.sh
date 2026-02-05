#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
TASK_IDS="${TASK_IDS:-0}"
MODEL="${MODEL:-qwen3-vl}"
INSTRUCTION_PATH="${INSTRUCTION_PATH:-agent/prompts/jsons/p_vision_ug_qwen.json}"

if [[ ! -f "$INSTRUCTION_PATH" ]]; then
  echo "Missing instruction file: $INSTRUCTION_PATH"
  echo "Run: python agent/prompts/to_json.py"
  exit 1
fi

export INSTRUCTION_PATH
export GROUNDING_MODEL="${GROUNDING_MODEL:-$MODEL}"

echo "=== Running with UGround grounding ==="
GROUNDING_METHOD=uground ./scripts/test_webcanvas.sh "$PORT" "$TASK_IDS" "grounding_uground" "$MODEL"

echo "=== Running with SIMPLE grounding ==="
GROUNDING_METHOD=simple ./scripts/test_webcanvas.sh "$PORT" "$TASK_IDS" "grounding_simple" "$MODEL"
