#!/usr/bin/env bash
set -e

# RubricHub data synthesis entrypoint (beginner friendly)
#
# Usage:
#   1) Edit the values in "1) Fill here"
#   2) Run: ./run_data_synthesis.sh

# Go to repo root so relative paths work.
cd "$(dirname "$0")"

###############
# 1) Fill here
###############

# Input data (JSONL) + which column is the question/query
INPUT_JSONL="local/your_input.jsonl"
QUESTION_COLUMN="question"

# -------------------------
# Advanced (optional)
# -------------------------
# You do NOT need to provide these. The pipeline only requires the query column.
# Set them only if your input already has stable identifiers and you want to carry them through.
ID_COLUMN=""               # e.g. "id"

# Output directory (intermediate + final artifacts will be written here)
OUTPUT_DIR="local/data_synthesis_output"

# -------------------------
# Per-step endpoint config (required)
# -------------------------
# This pipeline is designed to allow different steps to use different providers/endpoints.
# Fill `*_BASE_URL` + `*_API_KEY` for each model slot.
# - If your endpoint doesn't require a key, set `*_API_KEY` to any non-empty string (e.g., "dummy").
#   (OpenAI SDK expects an API key string even if your server ignores it.)
REFERENCE_API_KEY=""
REFERENCE_BASE_URL=""

RESPONSE_A_API_KEY=""
RESPONSE_A_BASE_URL=""
RESPONSE_B_API_KEY=""
RESPONSE_B_BASE_URL=""

RUBRIC_A_API_KEY=""
RUBRIC_A_BASE_URL=""
RUBRIC_B_API_KEY=""
RUBRIC_B_BASE_URL=""

MERGE_API_KEY=""
MERGE_BASE_URL=""

AUGMENT_API_KEY=""
AUGMENT_BASE_URL=""

# -------------------------
# Models (Fixed design)
# -------------------------
# - ONE model generates the `reference` answer used for rubric grounding.
# - TWO models generate `response_a/response_b` used for difficulty evolution.
# Paper used frontier models such as GPT-5.1 / Gemini 3 Pro Preview.
# Fill these with the model names available on your endpoint(s).
REFERENCE_MODEL="gpt-5.1"
RESPONSE_MODEL_A="gpt-5.1"
# Paper mentions using heterogeneous frontier models (e.g., GPT-5.1 + Gemini 3 Pro Preview).
# If you don't have Gemini on your endpoint, set this to the same as RESPONSE_MODEL_A.
RESPONSE_MODEL_B="gemini-3-pro-preview"

# Candidate rubric generation models (multi-model aggregation; paper default uses >=2 heterogeneous models)
RUBRIC_MODEL_A="gpt-5.1"
# Paper Stage2 explicitly uses heterogeneous frontier models to reduce single-model bias (e.g., GPT-5.1 + Gemini 3 Pro Preview).
# If you only have ONE model/provider, set this to the same as RUBRIC_MODEL_A (or leave empty) and the pipeline will degrade to single-model generation.
RUBRIC_MODEL_B="gemini-3-pro-preview"

# Rubric aggregation + difficulty evolution models
MERGE_MODEL="gpt-5.1"
AUGMENT_MODEL="gpt-5.1"

# -------------------------
# Runtime knobs
# -------------------------
CONCURRENCY="256"
RETRY="5"
TIMEOUT="1200"

# Paper alignment (Appendix B / Table 5):
# - RuRL rollouts: temperature=1.0, max response length=8192
# In this open-source synthesis pipeline we keep rubric-related steps low-temp for stability,
# and use higher temperature only for Step1 response sampling.
TEMPERATURE="0.2"
RESPONSE_TEMPERATURE="1.0"
SAVE_EVERY="200"

REF_MAX_TOKENS="8192"
RESPONSE_MAX_TOKENS="8192"
RUBRIC_MAX_TOKENS="2048"
MERGE_MAX_TOKENS="2048"
AUGMENT_MAX_TOKENS="2048"

# Optional: cap the number of final criteria per sample (0 = no cap)
MAX_CRITERIA="0"

###############
# 2) Run
###############

if [ ! -f "$INPUT_JSONL" ]; then
  echo "Input JSONL not found: $INPUT_JSONL" >&2
  echo "Set INPUT_JSONL to a valid path (relative to repo root)." >&2
  exit 1
fi

python data_synthesis_final/run_pipeline.py \
  --input "$INPUT_JSONL" \
  --question-column "$QUESTION_COLUMN" \
  --id-column "$ID_COLUMN" \
  --output-dir "$OUTPUT_DIR" \
  --reference-api-key "$REFERENCE_API_KEY" \
  --reference-base-url "$REFERENCE_BASE_URL" \
  --response-api-key-a "$RESPONSE_A_API_KEY" \
  --response-base-url-a "$RESPONSE_A_BASE_URL" \
  --response-api-key-b "$RESPONSE_B_API_KEY" \
  --response-base-url-b "$RESPONSE_B_BASE_URL" \
  --rubric-api-key-a "$RUBRIC_A_API_KEY" \
  --rubric-base-url-a "$RUBRIC_A_BASE_URL" \
  --rubric-api-key-b "$RUBRIC_B_API_KEY" \
  --rubric-base-url-b "$RUBRIC_B_BASE_URL" \
  --merge-api-key "$MERGE_API_KEY" \
  --merge-base-url "$MERGE_BASE_URL" \
  --augment-api-key "$AUGMENT_API_KEY" \
  --augment-base-url "$AUGMENT_BASE_URL" \
  --reference-model "$REFERENCE_MODEL" \
  --response-model-a "$RESPONSE_MODEL_A" \
  --response-model-b "$RESPONSE_MODEL_B" \
  --rubric-model-a "$RUBRIC_MODEL_A" \
  --rubric-model-b "$RUBRIC_MODEL_B" \
  --merge-model "$MERGE_MODEL" \
  --augment-model "$AUGMENT_MODEL" \
  --concurrency "$CONCURRENCY" \
  --retry "$RETRY" \
  --timeout "$TIMEOUT" \
  --temperature "$TEMPERATURE" \
  --response-temperature "$RESPONSE_TEMPERATURE" \
  --save-every "$SAVE_EVERY" \
  --ref-max-tokens "$REF_MAX_TOKENS" \
  --response-max-tokens "$RESPONSE_MAX_TOKENS" \
  --rubric-max-tokens "$RUBRIC_MAX_TOKENS" \
  --merge-max-tokens "$MERGE_MAX_TOKENS" \
  --augment-max-tokens "$AUGMENT_MAX_TOKENS" \
  --max-criteria "$MAX_CRITERIA"
