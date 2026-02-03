from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from step0_generate_reference import generate_references
from step1_generate_responses import generate_responses
from step2_generate_rubrics import generate_rubrics
from step3_merge_rubrics import merge_rubrics
from step4_augment_rubrics import augment_rubrics
from step5_export_dataset import export_dataset


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the full RubricHub data synthesis pipeline (resume supported)")
    p.add_argument("--input", required=True, help="Raw input JSONL")
    p.add_argument("--question-column", default="question", help="Which column is the query/question in input JSONL")
    p.add_argument("--id-column", default=None, help="Optional id column (used for resume key)")
    p.add_argument("--output-dir", required=True, help="Directory to write intermediate + final outputs")

    p.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    p.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base_url")
    p.add_argument("--reference-api-key", default=None, help="Optional API key override for reference model")
    p.add_argument("--reference-base-url", default=None, help="Optional base_url override for reference model")
    p.add_argument("--response-api-key-a", default=None, help="Optional API key override for response model A")
    p.add_argument("--response-base-url-a", default=None, help="Optional base_url override for response model A")
    p.add_argument("--response-api-key-b", default=None, help="Optional API key override for response model B")
    p.add_argument("--response-base-url-b", default=None, help="Optional base_url override for response model B")
    p.add_argument("--rubric-api-key-a", default=None, help="Optional API key override for rubric model A")
    p.add_argument("--rubric-base-url-a", default=None, help="Optional base_url override for rubric model A")
    p.add_argument("--rubric-api-key-b", default=None, help="Optional API key override for rubric model B")
    p.add_argument("--rubric-base-url-b", default=None, help="Optional base_url override for rubric model B")
    p.add_argument("--merge-api-key", default=None, help="Optional API key override for merge model")
    p.add_argument("--merge-base-url", default=None, help="Optional base_url override for merge model")
    p.add_argument("--augment-api-key", default=None, help="Optional API key override for augment model")
    p.add_argument("--augment-base-url", default=None, help="Optional base_url override for augment model")

    # Fixed design: one model generates reference, two models generate responses.
    p.add_argument("--reference-model", required=True)
    p.add_argument("--response-model-a", required=True)
    p.add_argument("--response-model-b", required=True)
    p.add_argument("--rubric-model-a", required=True)
    p.add_argument("--rubric-model-b", default=None)
    p.add_argument("--merge-model", required=True)
    p.add_argument("--augment-model", required=True)

    p.add_argument("--concurrency", type=int, default=256)
    p.add_argument("--retry", type=int, default=5)
    p.add_argument("--timeout", type=int, default=1200)

    p.add_argument("--ref-max-tokens", type=int, default=8192)
    p.add_argument("--response-max-tokens", type=int, default=8192)
    p.add_argument("--rubric-max-tokens", type=int, default=2048)
    p.add_argument("--merge-max-tokens", type=int, default=2048)
    p.add_argument("--augment-max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.2, help="Default temperature for steps other than Step1")
    p.add_argument("--response-temperature", type=float, default=1.0, help="Temperature for Step1 response sampling")
    p.add_argument("--save-every", type=int, default=200)

    p.add_argument("--max-criteria", type=int, default=None, help="Optional cap for final criteria count")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_dir = Path(__file__).parent / "prompts"

    def _pick(primary: Optional[str], fallback: Optional[str]) -> Optional[str]:
        if primary is None:
            return fallback
        if isinstance(primary, str) and primary.strip() == "":
            return fallback
        return primary

    api_key_global = args.api_key
    base_url_global = args.base_url

    reference_api_key = _pick(args.reference_api_key, api_key_global)
    reference_base_url = _pick(args.reference_base_url, base_url_global)

    response_api_key_a = _pick(args.response_api_key_a, api_key_global)
    response_base_url_a = _pick(args.response_base_url_a, base_url_global)
    response_api_key_b = _pick(args.response_api_key_b, api_key_global)
    response_base_url_b = _pick(args.response_base_url_b, base_url_global)

    rubric_api_key_a = _pick(args.rubric_api_key_a, api_key_global)
    rubric_base_url_a = _pick(args.rubric_base_url_a, base_url_global)
    rubric_api_key_b = _pick(args.rubric_api_key_b, api_key_global)
    rubric_base_url_b = _pick(args.rubric_base_url_b, base_url_global)

    merge_api_key = _pick(args.merge_api_key, api_key_global)
    merge_base_url = _pick(args.merge_base_url, base_url_global)

    augment_api_key = _pick(args.augment_api_key, api_key_global)
    augment_base_url = _pick(args.augment_base_url, base_url_global)

    step0_out = out_dir / "step0_reference.jsonl"
    step1_out = out_dir / "step1_responses.jsonl"
    step2_out = out_dir / "step2_rubrics.jsonl"
    step3_out = out_dir / "step3_merged.jsonl"
    step4_out = out_dir / "step4_augmented.jsonl"
    final_jsonl = out_dir / "final.jsonl"
    final_parquet = out_dir / "final.parquet"

    import asyncio

    asyncio.run(
        generate_references(
            input_path=Path(args.input),
            output_path=step0_out,
            prompt_template_path=prompts_dir / "reference.txt",
            question_column=args.question_column,
            id_column=args.id_column,
            api_key=reference_api_key,
            base_url=reference_base_url,
            model=args.reference_model,
            max_tokens=args.ref_max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            retry=args.retry,
            timeout_s=args.timeout,
            save_every=args.save_every,
        )
    )

    asyncio.run(
        generate_responses(
            input_path=step0_out,
            output_path=step1_out,
            prompt_template_path=prompts_dir / "reference.txt",
            api_key=None,
            base_url=None,
            api_key_a=response_api_key_a,
            base_url_a=response_base_url_a,
            api_key_b=response_api_key_b,
            base_url_b=response_base_url_b,
            model_a=args.response_model_a,
            model_b=args.response_model_b,
            max_tokens=args.response_max_tokens,
            temperature=args.response_temperature,
            concurrency=args.concurrency,
            retry=args.retry,
            timeout_s=args.timeout,
            save_every=args.save_every,
        )
    )

    asyncio.run(
        generate_rubrics(
            input_path=step1_out,
            output_path=step2_out,
            prompt_template_path=prompts_dir / "rubric_generation.txt",
            api_key=None,
            base_url=None,
            api_key_a=rubric_api_key_a,
            base_url_a=rubric_base_url_a,
            api_key_b=rubric_api_key_b,
            base_url_b=rubric_base_url_b,
            model_a=args.rubric_model_a,
            model_b=args.rubric_model_b,
            max_tokens=args.rubric_max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            retry=args.retry,
            timeout_s=args.timeout,
            save_every=args.save_every,
            reference_field="reference",
        )
    )

    asyncio.run(
        merge_rubrics(
            input_path=step2_out,
            output_path=step3_out,
            prompt_template_path=prompts_dir / "rubric_aggregation.txt",
            api_key=merge_api_key,
            base_url=merge_base_url,
            merge_model=args.merge_model,
            max_tokens=args.merge_max_tokens,
            temperature=args.temperature,
            concurrency=max(1, args.concurrency // 2),
            retry=args.retry,
            timeout_s=args.timeout,
            save_every=args.save_every,
            rubrics_a_field="rubrics_a",
            rubrics_b_field="rubrics_b",
        )
    )

    asyncio.run(
        augment_rubrics(
            input_path=step3_out,
            output_path=step4_out,
            prompt_template_path=prompts_dir / "difficulty_evolution.txt",
            api_key=augment_api_key,
            base_url=augment_base_url,
            augment_model=args.augment_model,
            max_tokens=args.augment_max_tokens,
            temperature=args.temperature,
            concurrency=max(1, args.concurrency // 2),
            retry=args.retry,
            timeout_s=args.timeout,
            save_every=args.save_every,
            merged_field="merged_rubrics",
            response1_field="response_a",
            response2_field="response_b",
        )
    )

    export_dataset(
        input_path=step4_out,
        output_jsonl=final_jsonl,
        output_parquet=final_parquet,
        base_field="merged_rubrics",
        augment_field="augmented_rubrics",
        max_criteria=args.max_criteria,
        skip_missing=True,
    )


if __name__ == "__main__":
    main()
