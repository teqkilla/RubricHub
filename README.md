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
*   **[2026-01-19]** 🔥 **Code and Data Coming Soon!** We are preparing the release of the **data synthesis code** and the **post-training (RuFT & RuRL) code**. Stay tuned!
*   **[2026-01-17]** RubricHub dataser is released, see https://huggingface.co/datasets/sojuL/RubricHub_v1.
*   **[2026-01-12]** RubricHub paper is released, see https://arxiv.org/abs/2601.08430.
  

## 📖 Introduction
Reinforcement Learning with Verifiable Rewards (RLVR) has shown great success in math and coding. However, open-ended generation remains challenging due to the lack of ground truth.

We introduce **RubricHub**, a large-scale (**~110k**) and multi-domain rubric dataset constructed via an automated **Coarse-to-Fine Rubric Generation** framework. By synergizing principle-guided synthesis, multi-model aggregation, and difficulty evolution, our approach produces highly discriminative criteria capable of capturing subtle nuances in model responses.

Based on RubricHub, we propose a two-stage post-training pipeline:
1.  **RuFT (Rubric-based Rejection Sampling Fine-Tuning)**
2.  **RuRL (Rubric-based Reinforcement Learning)**

Experimental results show that our post-trained **Qwen3-14B** achieves **SOTA results on HealthBench (69.3)**, surpassing proprietary frontier models such as **GPT-5**.

## 🚀 Methodology

### Automated Coarse-to-Fine Rubric Generation
Existing rubrics often suffer from scalability bottlenecks and low discriminability. Our framework addresses this through three stages:

1.  **Principle-Guided & Response-Grounded Generation:** Synthesizing criteria anchored to specific response contexts and guided by meta-principles to prevent generic or hallucinatory criteria.
2.  **Multi-Model Aggregation:** Aggregating perspectives from heterogeneous frontier models (e.g., GPT-5.1, Gemini 3 Pro) to eliminate single-source bias.
3.  **Difficulty Evolution:** Evolving criteria to capture discriminative nuances between "excellent" and "exceptional" responses, preventing score saturation.

![Pipeline](assets/pipeline.png)
*(Note: You can upload Figure 2 from the paper to an `assets` folder)*

## 📊 RubricHub Dataset
RubricHub contains approximately **110k** high-quality query-rubric pairs across five major domains:

*   **🏥 Medical:** 27.1% (e.g., HealthBench, LLMEval-Med)
*   **🔬 Science:** 27.1% (e.g., ResearchQA, GPQA)
*   **📝 Instruction Following:** 20.9% (e.g., IFEval)
*   **✍️ Writing:** 15.9% (e.g., WritingBench)
*   **💬 Chat:** 9.0% (e.g., Arena-Hard)

The dataset features high-density supervision, with complex domains like Writing and Medical averaging over 30 fine-grained criteria per query.

## 📈 Experiments

We validated RubricHub using Qwen3 base models. The results demonstrate significant improvements across all domains.

**Key Result:** On **HealthBench**, our Qwen3-14B (post-trained with RuFT → RuRL) achieves a score of **69.3**, outperforming **GPT-5 (67.2)**.


## 🛠️ Usage

### Installation
*(Coming Soon)*

### Data Synthesis
The code for the **Coarse-to-Fine Rubric Generation** pipeline will be released here. This will allow users to generate high-quality rubrics for their own datasets.

### Training (RuFT & RuRL)
We will provide scripts to reproduce our post-training pipeline using the `RubricHub` dataset:
1.  **RuFT:** Rejection sampling using rubric scores as filters.
2.  **RuRL:** Reinforcement learning using rubric scores as dense rewards (built on `verl` framework).

## 🖊️ Citation

If you find RubricHub useful for your research, please cite our paper:

```bibtex
@article{li2026rubrichub,
  title={RubricHub: A Comprehensive and Highly Discriminative Rubric Dataset via Automated Coarse-to-Fine Generation},
  author={Li, Sunzhu and Zhao, Jiale and Wei, Miteto and Ren, Huimin and Zhou, Yang and Yang, Jingwen and Liu, Shunyu and Zhang, Kaike and Chen, Wei},
  journal={arXiv preprint},
  year={2026}
}
```

## 📄 License
This project is licensed under the Apache 2.0 License.
