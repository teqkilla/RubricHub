# RubricHub Data Synthesis（Coarse-to-Fine Rubric Generation）

这份目录是 RubricHub 论文中 **Coarse-to-Fine Rubric Generation**（粗到细 rubric 合成）的开源实现。

- **入口脚本**：`./run_data_synthesis.sh`（编辑顶部 “1) Fill here”，然后直接运行）
- **输入**：一份 JSONL（每行一个 JSON object），只要求有一列是用户输入文本（query/question/prompt，列名可配）
- **输出**：`$OUTPUT_DIR/final.parquet`（主产物）与 `$OUTPUT_DIR/final.jsonl`（便于查看）
- **中间产物**：`step0_reference.jsonl` ~ `step4_augmented.jsonl`（用于断点重续/排错）

---

## 架构说明（给维护者/未来的我）

### 这段代码在论文里的位置：从 RubricHub 到 RL 训练信号

论文的核心链路可以分成两段：

1) **Coarse-to-Fine Rubric Generation（构建 RubricHub）**  
   对每个 query 自动合成一套高密度、强可区分（highly discriminative）的 rubric criteria。
2) **Rubrics in Post-Training（把 rubric 变成训练信号）**  
   - **RuFT**（Rubric-based Rejection Sampling Fine-Tuning）：用 rubric 打分过滤/挑选高质量回答，形成 SFT 数据。  
   - **RuRL**（Rubric-based Reinforcement Learning）：把 rubric 当作 dense reward 的“可验证/可判定”结构化奖励，对 policy 做 RL。

本仓库当前开源出来的代码只实现了第 (1) 段：**把“只有 query 的 JSONL”合成“(query, rubrics)”数据**。  
第 (2) 段（grader + RuFT/RuRL 训练）在论文里有完整定义，但不在这份 `data_synthesis_final/` 代码里。

### 方法与代码的一一对应（Coarse-to-Fine 三阶段 → Step0~Step4）

论文里的三阶段：

- **Stage 1: Response-Grounded & Principle-Guided Generation**  
  通过参考回答（reference response）把 rubric 生成“锚定”到具体上下文；同时用 meta-principles 约束 rubric 质量，减少 rubric drift。
- **Stage 2: Multi-Model Aggregation**  
  多模型并行生成候选 rubric，然后做保守合并/去重，减少单一模型偏差（perspective bias）。
- **Stage 3: Difficulty Evolution**  
  基于高质量回答对的差异，生成更严格、更挑剔、更能拉开差距的新增 criteria，避免评分饱和（score saturation）。

代码实现映射（`data_synthesis_final/`）：

```
raw.jsonl
  └─ Step0 生成 reference（response-grounding 的锚）
      └─ step0_reference.jsonl
          └─ Step1 生成 response_a/response_b（给 difficulty evolution 提供对比样本）
              └─ step1_responses.jsonl
                  └─ Step2 生成 rubrics_a/rubrics_b（principle-guided & response-grounded）
                      └─ step2_rubrics.jsonl
                          └─ Step3 merge → merged_rubrics（multi-model aggregation）
                              └─ step3_merged.jsonl
                                  └─ Step4 augment → augmented_rubrics（difficulty evolution）
                                      └─ step4_augmented.jsonl
                                          └─ Step5 export → final.{jsonl,parquet}
```

> 注意：论文里的“选取高质量参考回答对 A_ref”通常来自更大的候选池 + rubric 打分共识；  
> 这里的开源实现为了可跑性，直接用两个模型生成 `response_a/response_b` 作为对比输入。

### 固定设计（这个 repo 里被“定死”的选择）

这些选择在 `run_data_synthesis.sh` 和 `data_synthesis_final/run_pipeline.py` 里是硬约束（或“为了能跑通”的可选降级）：

- **reference 只生成 1 遍**：`REFERENCE_MODEL` → 字段 `reference`
- **responses 固定 2 路**：`RESPONSE_MODEL_A/B` → 字段 `response_a/response_b`
- **candidate rubrics 支持 1~2 路**：`RUBRIC_MODEL_A` 必填，`RUBRIC_MODEL_B` 可选 → `rubrics_a/rubrics_b`
- **（论文默认）candidate rubrics 用多模型**：论文的 Stage 2 是 **Multi-Model Aggregation**，核心就是用 >=2 个异构模型生成候选 rubric，再做聚合以降低单模型偏差。  
  这份开源实现为了“只有一个 endpoint 也能跑通”，允许 `RUBRIC_MODEL_B` 为空；但想贴论文复现/拿到更稳的 rubrics，建议一定配置 `RUBRIC_MODEL_B` 并指向不同 provider（配套 `RUBRIC_B_BASE_URL`）。
- **merge/augment 各 1 个模型**：`MERGE_MODEL` / `AUGMENT_MODEL`

### 目录结构与职责（按“入口 → 编排 → 工具 → 每步逻辑”理解）

- `run_data_synthesis.sh`  
  给外部用户唯一暴露的入口：用环境变量/脚本变量配置输入/输出/模型/并发，然后调用 `python data_synthesis_final/run_pipeline.py ...`。

- `data_synthesis_final/run_pipeline.py`  
  Pipeline orchestrator：串联 Step0→Step5，约定中间产物文件名，向各 step 透传运行参数。

- `data_synthesis_final/common.py`  
  Shared utilities（这份代码的“骨架”）：
  - JSONL 读写：`read_jsonl()` / `write_jsonl()`（用 `.tmp` 原子替换，避免中断损坏）
  - Resume 索引：`index_existing_rows()` + `__record_id__` 规则
  - OpenAI Async client：`make_async_client()`
  - 通用重试：`chat_completion_with_retry()`（RateLimit/Timeout/5xx 指数退避）
  - JSON 提取：`extract_json_array()`（从 ```json 代码块/原始文本里抓数组）
  - rubric 清洗：`normalize_rubric_items()`（裁剪字段、weight clamp 到 0~10）
  - 导出前处理：`rubrics_to_final_format()` / `dedup_final_rubrics()`（按 criterion 去重保留更高分）

- `data_synthesis_final/step*.py`  
  每一步一个纯 Python 文件（均支持 resume）。

- `data_synthesis_final/prompts/*.txt`  
  Prompt 模板（策略主要写在这里，不要在代码里硬编码 prompt）。

### 每步输入/输出字段（你调试/扩展时最常用的信息）

所有步骤都会保留并传递这些“元字段”：

- `question`: 规范化后的 query（最终导出字段）
- `id`: 可选透传（为空字符串也允许）
- `__source_index__`: 原始输入行号（Step0 生成）
- `__record_id__`: resume 主键（优先 `ID_COLUMN`，否则 `idx:<row_index>`）

Step0 `step0_generate_reference.py`

- 输入：raw JSONL（只要 `QUESTION_COLUMN`）
- 输出新增字段：
  - `reference`: reference answer（纯文本）
  - `reference_model`
  - `reference_error`（失败时）

Step1 `step1_generate_responses.py`

- 输入：Step0 输出
- 输出新增字段：
  - `response_a`, `response_a_model`, `response_a_error`
  - `response_b`, `response_b_model`, `response_b_error`

Step2 `step2_generate_rubrics.py`

- 输入：Step1 输出（依赖 `reference`）
- 输出新增字段：
  - `rubrics_a`: list[{title, description, weight}]
  - `rubrics_b`: 同上（可选）
  - `rubrics_a_model` / `rubrics_b_model`
  - `rubrics_a_error` / `rubrics_b_error`

Step3 `step3_merge_rubrics.py`

- 输入：Step2 输出（依赖 `rubrics_a`；`rubrics_b` 可选）
- 输出新增字段：
  - `merged_rubrics`: list[{title, description, weight}]
  - `merged_rubrics_model`
  - `merged_rubrics_error`
  - 若 `rubrics_b` 为空：直接 passthrough `rubrics_a`，并将 model 标为 `"passthrough"`

Step4 `step4_augment_rubrics.py`

- 输入：Step3 输出（依赖 `merged_rubrics` + `response_a/response_b`）
- 输出新增字段：
  - `augmented_rubrics`: list[{title, description, weight}]（只应包含“新增更严格条目”）
  - `augmented_rubrics_model`
  - `augmented_rubrics_error`
  - 若缺少 responses：会输出空列表并标记 model 为 `"skipped(no responses)"`（这被视作合法输出）

Step5 `step5_export_dataset.py`

- 输入：Step4 输出
- 输出：
  - `final.jsonl`：`{question, id, rubrics:[{criterion, points}]}`
  - `final.parquet`：同 schema，但 `rubrics` 是嵌套列 `list<struct<criterion: string, points: int32>>`

导出规则（非常关键）：

- 先拼接 `merged_rubrics + augmented_rubrics`
- 将 `{description, weight}` 映射为 `{criterion, points}`
- 以规范化后的 `criterion` 文本去重：保留 **points 更高** 的那条
- 可选：`MAX_CRITERIA` / `--max-criteria` 截断最终 criteria 数量

### Resume（断点重续）机制：为什么能“反复跑不重复花钱”

核心逻辑在各 step 的这套模式：

1) 读取本 step 的 `output_path`（已有中间产物）
2) 用 `index_existing_rows()` 建立 `__record_id__ → row` 映射
3) 对每条输入：
   - 如果输出里对应字段存在且没有 `*_error`：直接复用
   - 否则：进入 pending 队列重跑，并把异常写回 `*_error`

`__record_id__` 的生成规则（Step0 执行）：

- 如果你提供了 `ID_COLUMN` 且该字段非空：用它当主键（推荐，跨数据变更更稳定）
- 否则：使用行号生成 `idx:<n>`（输入行顺序一变就会导致 resume miss）

### LLM 调用：并发、重试、以及输出为什么必须是 JSON

- 所有 LLM 调用走 OpenAI Python SDK 的 async client：`client.chat.completions.create(...)`
- `chat_completion_with_retry()` 对 RateLimit/Timeout/连接错误/5xx 做指数退避重试
- `run_with_concurrency()` 用 async worker queue 控制并发，并定期写盘（`SAVE_EVERY`）

参数与论文对齐说明：

- 论文 Appendix B / Table 5 明确给了 **RuRL rollout** 的采样超参：`temperature=1.0`、`Max Response Length=8192`。  
  这份开源 pipeline 虽然不包含 RL 训练，但 **Step1 的 response 采样**沿用这个设定（见 `run_data_synthesis.sh` 的 `RESPONSE_TEMPERATURE` / `RESPONSE_MAX_TOKENS`）。
- 论文没有明确写 rubric 合成阶段（reference/rubric/merge/augment）的温度与 max_tokens；这里默认把这些步骤设为较低温度以提高稳定性和可复现性。

Step2~Step4 的输出要求是 **“严格 JSON array”**，原因：

- `extract_json_array()` 会从输出里提取 `[...]` 并 `json.loads()`  
  一旦模型输出混入多余解释文字、或不是数组结构，就会触发 error，写入 `*_error` 并在下次 resume 重跑。

### Prompt 模板（修改策略优先改这里）

占位符约定：

- `reference.txt`：`<<query>>`
- `rubric_generation.txt`：`<<query>>` + `<<reference>>`
- `rubric_aggregation.txt`：`{|prompt|}` + `{|rubrics1|}` + `{|rubrics2|}`
- `difficulty_evolution.txt`：`{|prompt|}` + `{|rubrics|}` + `{|response1|}` + `{|response2|}`

修改策略建议：

- 想更“挑剔/可区分”：优先改 `difficulty_evolution.txt`（强调 discriminative、二值判定、避免双方都满足）
- 想更“贴题/不漂”：优先改 `rubric_generation.txt`（强调 response-grounding 与覆盖显式/隐式要求）
- 想更“保守去重”：优先改 `rubric_aggregation.txt`（merge only if identical）

### Rubric 质量原则（来自论文 Appendix A / Table 4）

论文把“高质量 rubric”拆成四类 meta-dimensions（这里用人话翻译一下，方便你对齐 prompt 是否覆盖）：

- **Consistency & Alignment**：rubric 要能让不同 grader 给出一致结论；每条 criterion 必须与 query 相关  
- **Structure & Scope**：覆盖显式/隐式要求；criterion 数量控制在合理范围；各条 criterion 相互独立/不重叠，且尽量原子化  
- **Clarity & Quality**：表述清晰、避免模糊词；长度适中；语言与 query 一致  
- **Reasoning & Evaluability**：能区分模型能力；权重合理；criterion 可通过可观察证据验证（verifiable / checkable）

在这份开源实现里，这些原则主要体现在 `rubric_generation.txt` / `rubric_aggregation.txt` / `difficulty_evolution.txt` 的规则约束里。

### 与论文“完整版”实现的差异（有意的简化点）

- **权重范围**：论文里 weight 可为负（penalty-based）且范围更宽；这里为简化导出与去重，统一 clamp 到 `0~10`（见 `common.py`）。  
- **A_ref 选择**：论文里 difficulty evolution 的参考回答对通常来自“候选池 + rubric 共识高分”的筛选；这里直接用两路模型生成的 `response_a/response_b`。  
- **grader/奖励**：论文里 RuRL 依赖 grader 把 criterion 判成二值 `b_i`；本 repo 不包含 grader 与 RL 训练实现，只生成 rubrics 数据本身。  
- **模型组合**：论文的 multi-model aggregation 往往用更多异构模型；这里默认 1~2 个 rubric 生成模型 + 1 个 merge 模型完成 aggregation。

### 与 RuFT / RuRL 的衔接（论文里 RL 训练数据怎么来）

这份代码导出的 `final.parquet` 本质上就是论文里的 **(q, R_q)**。
接下来的训练数据/奖励信号（不在本 repo）一般这样接：

- **RuFT（构建 SFT 数据）**  
  对每个 (q, R_q) 生成 K 个候选回答 `A={a_k}`，用 rubric 打分 `S_k`（按 criterion weight 聚合并归一化），
  过滤低于阈值 `τ` 的回答，取最高分 `a+`，得到 SFT 对 `(q, a+)`。

- **RuRL（把 rubric 当 reward）**  
  对每条 criterion `c_i`，grader 输出二值 `b_i∈{0,1}`（可用规则或 LLM grader），
  reward 用加权和归一化：`r = (Σ w_i b_i) / (Σ w_i)`，再用 RL 算法优化 policy。

> 论文附录中还给了 RuFT/RuRL 的具体超参（如阈值 τ、候选数、DAPO 配置等）；  
> 这里不复现训练代码，但这段衔接解释能帮助你把 `final.parquet` 对接到后续训练系统。
