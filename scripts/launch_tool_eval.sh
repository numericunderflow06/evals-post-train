#!/usr/bin/env bash
set -euo pipefail

#
# Launch tool-use benchmark evaluations
#
# Usage:
#   bash scripts/launch_tool_eval.sh tau2-bench --model gpt-4o --backend openai
#   bash scripts/launch_tool_eval.sh swe-bench --model claude-sonnet-4-20250514 --backend anthropic --limit 50
#   bash scripts/launch_tool_eval.sh livecodebench --model Qwen/Qwen2.5-72B-Instruct --backend vllm
#
# Benchmarks:
#   tau2-bench     - Conversational agent tool use (airline, retail, telecom)
#   swe-bench      - GitHub issue resolution
#   livecodebench  - Competitive programming with code execution
#   terminalbench  - Terminal-based tasks
#   browsecomp     - Web browsing and information retrieval
#   mcp-atlas      - MCP multi-server tool orchestration
#

VALID_BENCHMARKS=("tau2-bench" "swe-bench" "livecodebench" "terminalbench" "browsecomp" "mcp-atlas")

# --- Parse arguments ---
BENCHMARK="${1:?Usage: $0 <benchmark> [options]}"
shift

MODEL=""
BACKEND="openai"
LIMIT=""
DOMAIN="retail"
TEMPERATURE="0.0"
MAX_TOKENS="4096"
MAX_TURNS=""
OUTPUT_DIR="./tool_eval_results"
WANDB_PROJECT=""
WANDB_ENTITY="apertus"
BASE_URL=""
API_KEY=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --domain) DOMAIN="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --max-turns) MAX_TURNS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
        --base-url) BASE_URL="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --verbose) VERBOSE="--verbose"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- Validate ---
if [[ ! " ${VALID_BENCHMARKS[*]} " =~ " ${BENCHMARK} " ]]; then
    echo "Invalid benchmark: ${BENCHMARK}"
    echo "Valid benchmarks: ${VALID_BENCHMARKS[*]}"
    exit 1
fi

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    exit 1
fi

# --- Build command ---
CMD="python -m tool_eval.evaluate"
CMD+=" --benchmark ${BENCHMARK}"
CMD+=" --backend ${BACKEND}"
CMD+=" --model ${MODEL}"
CMD+=" --temperature ${TEMPERATURE}"
CMD+=" --max-tokens ${MAX_TOKENS}"
CMD+=" --output-dir ${OUTPUT_DIR}"

[[ -n "$LIMIT" ]] && CMD+=" --limit ${LIMIT}"
[[ -n "$MAX_TURNS" ]] && CMD+=" --max-turns ${MAX_TURNS}"
[[ -n "$BASE_URL" ]] && CMD+=" --base-url ${BASE_URL}"
[[ -n "$API_KEY" ]] && CMD+=" --api-key ${API_KEY}"
[[ -n "$VERBOSE" ]] && CMD+=" ${VERBOSE}"

if [[ "$BENCHMARK" == "tau2-bench" ]]; then
    CMD+=" --domain ${DOMAIN}"
fi

if [[ -n "$WANDB_PROJECT" ]]; then
    CMD+=" --wandb-project ${WANDB_PROJECT}"
    CMD+=" --wandb-entity ${WANDB_ENTITY}"
fi

echo "============================================"
echo "Tool-Use Evaluation"
echo "============================================"
echo "Benchmark: ${BENCHMARK}"
echo "Model:     ${MODEL}"
echo "Backend:   ${BACKEND}"
echo "Command:   ${CMD}"
echo "============================================"

eval $CMD
