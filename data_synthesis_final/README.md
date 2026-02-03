# RubricHub Data Synthesis (Coarse-to-Fine Rubric Generation)

This directory contains the open-source implementation of RubricHub’s **Coarse-to-Fine Rubric Generation** pipeline.

- **Entrypoint**: `./run_data_synthesis.sh` (edit the top “1) Fill here”, then run)
- **Input**: a JSONL file (one JSON object per line) with one field containing the user prompt text (query/question/prompt; the column name is configurable)
- **Output**: `$OUTPUT_DIR/final.parquet` (main artifact) and `$OUTPUT_DIR/final.jsonl` (easy to inspect)
- **Intermediates**: `step0_reference.jsonl` ~ `step4_augmented.jsonl` (for resume/debug)

---

## Architecture Notes (for maintainers / future me)

### Where this code sits in the paper: from RubricHub to RL training signals

The paper’s overall pipeline has two parts:

1) **Coarse-to-Fine Rubric Generation (build RubricHub)**  
   For each query, automatically synthesize a high-density, highly-discriminative set of rubric criteria.
2) **Rubrics in Post-Training (turn rubrics into training signals)**  
   - **RuFT** (Rubric-based Rejection Sampling Fine-Tuning): score/filter candidate answers with rubrics to form SFT data.  
   - **RuRL** (Rubric-based Reinforcement Learning): treat rubric criteria as a structured, verifiable reward to run RL.

This repository currently implements **only (1)**: it turns “JSONL with only prompts” into “(prompt, rubrics)” data.  
Part (2) (grader + RuFT/RuRL training) is defined in the paper but is **not** implemented in `data_synthesis_final/`.

### Paper stages ↔ code steps (Stage 1–3 → Step0–Step4)

Paper stages:

- **Stage 1: Response-Grounded & Principle-Guided Generation**  
  Generate rubrics grounded in a reference response, constrained by meta-principles to reduce rubric drift.
- **Stage 2: Multi-Model Aggregation**  
  Generate candidate rubrics with multiple models, then conservatively merge/deduplicate to reduce single-model perspective bias.
- **Stage 3: Difficulty Evolution**  
  Given a pair of answers, evolve stricter and more discriminative criteria to avoid score saturation.

Implementation mapping (`data_synthesis_final/`):

```
raw.jsonl
  └─ Step0: generate reference (grounding anchor)
      └─ step0_reference.jsonl
          └─ Step1: generate response_a/response_b (contrast answers for difficulty evolution)
              └─ step1_responses.jsonl
                  └─ Step2: generate rubrics_a/rubrics_b (principle-guided & response-grounded)
                      └─ step2_rubrics.jsonl
                          └─ Step3: merge → merged_rubrics (multi-model aggregation)
                              └─ step3_merged.jsonl
                                  └─ Step4: augment → augmented_rubrics (difficulty evolution)
                                      └─ step4_augmented.jsonl
                                          └─ Step5: export → final.{jsonl,parquet}
```

Note: in the paper, the “high-quality answer pair” for difficulty evolution is typically selected from a larger candidate pool with rubric-scoring consensus.  
This open-source implementation keeps things runnable and uses two generated answers (`response_a/response_b`) directly.

### Fixed design choices (hard constraints in this repo)

These are “fixed” by `run_data_synthesis.sh` + `data_synthesis_final/run_pipeline.py` (or degraded in a controlled way for usability):

- **Reference generated once**: `REFERENCE_MODEL` → field `reference`
- **Two responses**: `RESPONSE_MODEL_A/B` → fields `response_a/response_b`
- **1–2 rubric generators**: `RUBRIC_MODEL_A` required, `RUBRIC_MODEL_B` optional → `rubrics_a/rubrics_b`
- **Paper-default is multi-model**: Stage 2 is **Multi-Model Aggregation**, so using >=2 heterogeneous models is recommended for stronger rubrics.  
  This repo allows `RUBRIC_MODEL_B` to be empty so it can still run with only one provider.
- **One merge model + one augment model**: `MERGE_MODEL` / `AUGMENT_MODEL`

### Directory responsibilities (entry → orchestration → shared utilities → step logic)

- `run_data_synthesis.sh`  
  The only user-facing entrypoint: configure input/output/models/endpoints/concurrency, then call `python data_synthesis_final/run_pipeline.py ...`.

- `data_synthesis_final/run_pipeline.py`  
  Pipeline orchestrator: wires Step0→Step5, defines intermediate filenames, passes runtime parameters to each step.

- `data_synthesis_final/common.py`  
  Shared utilities (the “skeleton”):
  - JSONL IO: `read_jsonl()` / `write_jsonl()` (atomic write via `.tmp` to avoid corruption on interruption)
  - Resume indexing: `index_existing_rows()` + `__record_id__` rules
  - OpenAI async client: `make_async_client()`
  - Retry wrapper: `chat_completion_with_retry()` (exponential backoff on rate limit / timeout / 5xx)
  - JSON extraction: `extract_json_array()` (pull `[...]` from fenced blocks or raw text)
  - Rubric normalization: `normalize_rubric_items()` (field trimming + clamp `weight` to 0–10)
  - Export helpers: `rubrics_to_final_format()` / `dedup_final_rubrics()` (dedup by criterion text, keep higher points)

- `data_synthesis_final/step*.py`  
  One file per step (each step supports resume).

- `data_synthesis_final/prompts/*.txt`  
  Prompt templates (edit prompts here; avoid hardcoding prompt text in Python).

### Per-step fields (most useful when debugging/extending)

All steps preserve and propagate these “meta fields”:

- `question`: normalized prompt text (final exported field)
- `id`: optional passthrough id (empty string is allowed)
- `__source_index__`: original row index from the raw input (created in Step0)
- `__record_id__`: resume primary key (prefer `ID_COLUMN`, otherwise `idx:<row_index>`)

Step0 `step0_generate_reference.py`

- Input: raw JSONL (only needs `QUESTION_COLUMN`)
- Output adds:
  - `reference` (text)
  - `reference_model`
  - `reference_error` (on failure)

Step1 `step1_generate_responses.py`

- Input: Step0 output
- Output adds:
  - `response_a`, `response_a_model`, `response_a_error`
  - `response_b`, `response_b_model`, `response_b_error`

Step2 `step2_generate_rubrics.py`

- Input: Step1 output (requires `reference`)
- Output adds:
  - `rubrics_a`: list[{title, description, weight}]
  - `rubrics_b`: same schema (optional)
  - `rubrics_a_model` / `rubrics_b_model`
  - `rubrics_a_error` / `rubrics_b_error`

Step3 `step3_merge_rubrics.py`

- Input: Step2 output (requires `rubrics_a`; `rubrics_b` optional)
- Output adds:
  - `merged_rubrics`: list[{title, description, weight}]
  - `merged_rubrics_model`
  - `merged_rubrics_error`
  - If `rubrics_b` is missing/empty: passthrough `rubrics_a` and set model to `"passthrough"`

Step4 `step4_augment_rubrics.py`

- Input: Step3 output (requires `merged_rubrics` + `response_a/response_b`)
- Output adds:
  - `augmented_rubrics`: list[{title, description, weight}] (should contain only *new* stricter items)
  - `augmented_rubrics_model`
  - `augmented_rubrics_error`
  - If responses are missing: output `[]` and set model to `"skipped(no responses)"` (treated as a valid output)

Step5 `step5_export_dataset.py`

- Input: Step4 output
- Output:
  - `final.jsonl`: `{question, id, rubrics:[{criterion, points}]}`
  - `final.parquet`: same schema, but `rubrics` is a nested list column (`list<struct<criterion: string, points: int32>>`)

Export rules (important):

- Concatenate `merged_rubrics + augmented_rubrics`
- Map `{description, weight}` → `{criterion, points}`
- Deduplicate by normalized `criterion` text: keep the entry with **higher points**
- Optional: cap final criteria count via `MAX_CRITERIA` / `--max-criteria`

### Resume (why you can re-run without paying twice)

Each step follows the same pattern:

1) Load the step’s output JSONL if it exists
2) Build a `__record_id__ → row` index
3) For each input row:
   - If the output already has the target fields and no `*_error`: reuse it
   - Otherwise: re-run and write errors back to `*_error`

`__record_id__` generation (in Step0):

- If `ID_COLUMN` is provided and the field is non-empty: use it (recommended; stable across shuffles)
- Else: use the input row index (`idx:<n>`) (changing input order will break resume alignment)

### LLM calls: concurrency, retries, and why outputs must be JSON

- All LLM calls use OpenAI Python SDK async client: `client.chat.completions.create(...)`
- `chat_completion_with_retry()` handles rate limits / timeouts / 5xx with exponential backoff
- `run_with_concurrency()` uses an async worker queue and periodically writes intermediate results (`SAVE_EVERY`)

Paper-alignment note:

- The paper’s Appendix B / Table 5 specifies RuRL rollout sampling hyperparams: **temperature=1.0** and **max response length=8192**.  
  This repo does not include RL training, but **Step1 response sampling** keeps that setting via `RESPONSE_TEMPERATURE` / `RESPONSE_MAX_TOKENS` in `run_data_synthesis.sh`.
- The paper does not explicitly specify temperatures / max_tokens for rubric synthesis steps (reference/rubric/merge/augment); the defaults here keep those steps low-temperature for stability.

Step2–Step4 require outputs to be a **strict JSON array**:

- `extract_json_array()` extracts the first valid `[...]` and runs `json.loads()`.
- Any extra explanation text or non-array output becomes an error and will be retried on resume.

### Prompt templates (edit strategy here first)

Placeholder conventions:

- `reference.txt`: `<<query>>`
- `rubric_generation.txt`: `<<query>>` + `<<reference>>`
- `rubric_aggregation.txt`: `{|prompt|}` + `{|rubrics1|}` + `{|rubrics2|}`
- `difficulty_evolution.txt`: `{|prompt|}` + `{|rubrics|}` + `{|response1|}` + `{|response2|}`

Editing tips:

- Want more “picky / discriminative” criteria: tweak `difficulty_evolution.txt`
- Want more “on-topic / less drifting”: tweak `rubric_generation.txt`
- Want more “conservative merging”: tweak `rubric_aggregation.txt` (merge only when clearly equivalent)

### Rubric quality principles (paper Appendix A / Table 4)

The paper groups “good rubric” properties into four meta dimensions:

- **Consistency & Alignment**: graders should agree; each criterion must be relevant to the query
- **Structure & Scope**: cover explicit/implicit requirements; reasonable criterion count; criteria should be atomic and non-overlapping
- **Clarity & Quality**: avoid vague wording; concise and clear; language matches the query
- **Reasoning & Evaluability**: criteria should be discriminative and verifiable/checkable; weights should be reasonable

In this implementation, these constraints primarily live in the prompt templates:
`rubric_generation.txt` / `rubric_aggregation.txt` / `difficulty_evolution.txt`.

### Intended simplifications vs the “full” paper system

- **Weight range**: the paper allows wider ranges (including negative penalty weights). For export simplicity, this repo clamps to `0–10` (see `common.py`).
- **Answer-pair selection**: the paper uses selection from a larger pool; this repo uses two generated answers directly.
- **No grader / no RL code**: this repo produces `(q, R_q)` data but does not implement the grader or RuFT/RuRL training.
- **Fewer models**: the paper often uses more heterogeneous models; this repo defaults to 1–2 rubric generators + 1 merge model for practicality.

### How this output connects to RuFT / RuRL (paper-level)

The exported `final.parquet` is essentially the paper’s **(q, R_q)**.
Downstream (not in this repo) typically looks like:

- **RuFT (SFT data construction)**  
  For each `(q, R_q)`, sample `K` candidate answers `A={a_k}`, score them with the rubric, filter by threshold `τ`,
  select a high-scoring `a+`, and form SFT pairs `(q, a+)`.

- **RuRL (rubric as reward)**  
  For each criterion `c_i`, a grader outputs binary `b_i∈{0,1}` (rules or an LLM grader),
  and reward is the normalized weighted sum: `r = (Σ w_i b_i) / (Σ w_i)`, then run RL to optimize the policy.

