# RubricHub: A Comprehensive and Highly Discriminative Rubric Dataset via Automated Coarse-to-Fine Generation

<div align="center">

<!-- You can add badges here later, e.g., ArXiv, License, HuggingFace -->
<a href="https://arxiv.org/abs/2601.08430">
    <img src="https://img.shields.io/badge/arXiv-2601.08430-b31b1b.svg"" alt="Paper"/>
</a>
<a href="https://huggingface.co/datasets/sojuL/RubricHub_v1">
    <img src="https://img.shields.io/badge/Data-HuggingFace-yellow" alt="Rubrichub"/>
</a>
<a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/>
</a>

</div>


## 📢 News
*   **[2026-02-03]** 🔥 **Training recipes (external) are now available.** This repo currently focuses on data synthesis and does **not** ship an end-to-end training framework. For RuRL, refer to [RuscaRL](https://github.com/IANNXANG/RuscaRL) (sync) or [verl-rubric](https://github.com/DoYangTan/verl-rubric) (async). For RuFT/SFT, refer to [LlamaFactory](https://github.com/hiyouga/LlamaFactory). Our RubricHub rule-based scorer/grader integration for RuRL (incl. instruction-following rules) is being cleaned up and will be released soon.
*   **[2026-02-03]** 🔥 **Data synthesis code released.** See `data_synthesis_final/README.md`. 
*   **[2026-01-17]** RubricHub dataset is released, see https://huggingface.co/datasets/sojuL/RubricHub_v1.
*   **[2026-01-12]** RubricHub paper is released, see https://arxiv.org/abs/2601.08430.


## 📖 Introduction
Reinforcement Learning with Verifiable Rewards (RLVR) has shown great success in math and coding. However, open-ended generation remains challenging due to the lack of ground truth.

We introduce **RubricHub**, a large-scale (**~110k**) and multi-domain rubric dataset constructed via an automated **Coarse-to-Fine Rubric Generation** framework. By synergizing principle-guided synthesis, multi-model aggregation, and difficulty evolution, our approach produces highly discriminative criteria capable of capturing subtle nuances in model responses.

Based on RubricHub, we propose a two-stage post-training pipeline:
1.  **RuFT (Rubric-based Rejection Sampling Fine-Tuning)**
2.  **RuRL (Rubric-based Reinforcement Learning)**

Experimental results show that our post-trained **Qwen3-14B** achieves **SOTA results on HealthBench (69.3)**, surpassing proprietary frontier models such as **GPT-5**.

## 🚀 Methodology
![Pipeline](image/method.png)

### Automated Coarse-to-Fine Rubric Generation
Existing rubrics often suffer from scalability bottlenecks and low discriminability. Our framework addresses this through three stages:

1.  **Principle-Guided & Response-Grounded Generation:** Synthesizing criteria anchored to specific response contexts and guided by meta-principles to prevent generic or hallucinatory criteria.
2.  **Multi-Model Aggregation:** Aggregating perspectives from heterogeneous frontier models (e.g., GPT-5.1, Gemini 3 Pro) to eliminate single-source bias.
3.  **Difficulty Evolution:** Evolving criteria to capture discriminative nuances between "excellent" and "exceptional" responses, preventing score saturation.



## 📊 RubricHub Dataset
![method](image/dataset.png)

RubricHub contains approximately **110k** high-quality query-rubric pairs across five major domains:
*   **🏥 Medical:** 27.1%
*   **🔬 Science:** 27.1% 
*   **📝 Instruction Following:** 
*   **✍️ Writing:** 15.9%
*   **💬 Chat:** 9.0%

The dataset features high-density supervision, with complex domains like Writing and Medical averaging over 30 fine-grained criteria per query.

## 📈 Experiments
![Pipeline](image/results.png)
We validated RubricHub using Qwen3 base models. The results demonstrate significant improvements across all domains.

**Key Result:** On **HealthBench**, our Qwen3-14B (post-trained with RuFT → RuRL) achieves a score of **69.3**, outperforming **GPT-5 (67.2)**.


## 🛠️ Usage
### Data Synthesis (Coarse-to-Fine Rubric Generation)

1) (Recommended) Create a clean env:

```bash
conda create -n rubrichub python=3.10 -y
conda activate rubrichub
```

2) Install deps for the data synthesis pipeline:

```bash
pip install -U openai tqdm pyarrow
```

3) Prepare an input JSONL, and set `QUESTION_COLUMN` to the field name that contains your prompt text (it can be `question`, `prompt`, `instruction`, etc.).

4) Edit `run_data_synthesis.sh` (top “1) Fill here”):
- fill input/output paths and `QUESTION_COLUMN`
- **for each model slot** (`REFERENCE_*`, `RESPONSE_*`, `RUBRIC_*`, `MERGE_*`, `AUGMENT_*`), fill its `*_BASE_URL`, `*_API_KEY`, and `*_MODEL`
  - if your OpenAI-compatible server ignores API keys, set `*_API_KEY` to **any non-empty string** (e.g., `"dummy"`) since the OpenAI SDK requires it

5) Run:

```bash
./run_data_synthesis.sh
```

Outputs will be written to `$OUTPUT_DIR/`:
- `final.parquet` (main artifact)
- `final.jsonl` (same content, easier to inspect)
- `step0_reference.jsonl` ~ `step4_augmented.jsonl` (intermediates for resume/debug)

For pipeline architecture and implementation details, see `data_synthesis_final/README.md`.

### Training (RuFT & RuRL)
This repo currently does **not** include the full post-training implementation (grader + training loop). You can still reproduce the RuFT/RuRL pipeline using existing open-source trainers:

- **RuFT (Rubric-based Rejection Sampling Fine-Tuning):** sample multiple candidate answers per query, score them with a rubric grader, keep high-scoring answers to form SFT pairs, then run SFT with [LlamaFactory](https://github.com/hiyouga/LlamaFactory).
- **RuRL (Rubric-based Reinforcement Learning):** turn each criterion into a binary decision `b_i∈{0,1}` (via a rules/LLM grader), aggregate reward as `r = (Σ w_i b_i) / (Σ w_i)`, then run RL with [RuscaRL](https://github.com/IANNXANG/RuscaRL) (sync) or [verl-rubric](https://github.com/DoYangTan/verl-rubric) (async).

For details on how `final.parquet` connects to RuFT/RuRL, see the section “与 RuFT / RuRL 的衔接” in `data_synthesis_final/README.md`.

## 🖊️ Citation

If you find RubricHub useful for your research, please cite our paper:

```bibtex
@article{li2026rubrichub,
  title={RubricHub: A Comprehensive and Highly Discriminative Rubric Dataset via Automated Coarse-to-Fine Generation},
  author={Li, Sunzhu and Zhao, Jiale and Wei, Miteto and Ren, Huimin and Zhou, Yang and Yang, Jingwen and Liu, Shunyu and Zhang, Kaike and Chen, Wei},
  journal={arXiv preprint arXiv:2601.08430},
  year={2026}
}
```

## 📄 License
This project is licensed under the Apache 2.0 License.
