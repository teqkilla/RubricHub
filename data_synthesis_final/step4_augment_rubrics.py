from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from common import (
    JSONDict,
    chat_completion_with_retry,
    extract_json_array,
    get_llm_config,
    index_existing_rows,
    is_missing_or_error,
    load_text,
    make_async_client,
    normalize_rubric_items,
    read_jsonl,
    render_template,
    run_with_concurrency,
    write_jsonl,
)


async def augment_rubrics(
    *,
    input_path: Path,
    output_path: Path,
    prompt_template_path: Path,
    api_key: Optional[str],
    base_url: Optional[str],
    augment_model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    retry: int,
    timeout_s: int,
    save_every: int,
    merged_field: str,
    response1_field: str,
    response2_field: str,
) -> None:
    template = load_text(prompt_template_path)
    input_rows = read_jsonl(input_path)
    existing_map = index_existing_rows(read_jsonl(output_path))

    def _is_done(existing: JSONDict) -> bool:
        # allow empty list as a valid output if no error exists
        if existing.get("augmented_rubrics_error"):
            return False
        return "augmented_rubrics" in existing

    results: List[Optional[JSONDict]] = [None] * len(input_rows)
    pending: List[tuple[int, JSONDict]] = []
    reused = 0
    for i, row in enumerate(input_rows):
        rid = str(row.get("__record_id__", f"idx:{i}"))
        exist = existing_map.get(rid)
        if exist and _is_done(exist):
            merged = dict(row)
            merged.update(exist)
            merged.pop("rubric_id", None)
            results[i] = merged
            reused += 1
        else:
            if exist:
                merged = dict(row)
                merged.update(exist)
                merged.pop("rubric_id", None)
                pending.append((i, merged))
            else:
                pending.append((i, row))

    cfg = get_llm_config(api_key, base_url, timeout_s)
    client = make_async_client(cfg)

    pbar = tqdm(total=len(pending), desc="step4: augment", ncols=120)

    async def _process(i: int, row: JSONDict) -> JSONDict:
        out = dict(row)
        out.pop("rubric_id", None)
        q = str(out.get("question", "")).strip()
        merged = out.get(merged_field, [])
        resp1 = str(out.get(response1_field, "")).strip()
        resp2 = str(out.get(response2_field, "")).strip()

        if not isinstance(merged, list) or len(merged) == 0:
            out["augmented_rubrics"] = []
            out["augmented_rubrics_error"] = f"missing {merged_field}"
            return out

        if not resp1 or not resp2:
            out["augmented_rubrics"] = []
            out.pop("augmented_rubrics_error", None)
            out["augmented_rubrics_model"] = "skipped(no responses)"
            return out

        prompt = render_template(
            template,
            {
                "{|prompt|}": q,
                "{|rubrics|}": json.dumps(merged, ensure_ascii=False),
                "{|response1|}": resp1,
                "{|response2|}": resp2,
            },
        )

        try:
            if is_missing_or_error(out, "augmented_rubrics"):
                text = await chat_completion_with_retry(
                    client,
                    model=augment_model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    retry=retry,
                )
                arr = extract_json_array(text)
                out["augmented_rubrics"] = normalize_rubric_items(arr)
                out.pop("augmented_rubrics_error", None)
            out["augmented_rubrics_model"] = augment_model
        except Exception as e:
            out["augmented_rubrics"] = out.get("augmented_rubrics", []) or []
            out["augmented_rubrics_error"] = str(e)[:500]
        return out

    def _save_partial(res: List[Optional[JSONDict]]) -> None:
        write_jsonl(output_path, [r for r in res if r is not None])

    def _progress(done: int, total: int) -> None:
        pbar.update(1)

    await run_with_concurrency(
        pending,
        concurrency=concurrency,
        process_fn=_process,
        save_every=save_every,
        save_partial_fn=_save_partial,
        results=results,
        progress_cb=_progress,
    )
    pbar.close()

    write_jsonl(output_path, [r for r in results if r is not None])
    await client.close()

    print(f"step4 done. reused={reused} processed={len(pending)} output={output_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step4: augment rubrics (difficulty evolution)")
    p.add_argument("--input", required=True, help="Input JSONL (from step2)")
    p.add_argument("--output", required=True, help="Output JSONL")
    p.add_argument("--prompt-template", required=True, help="Difficulty-evolution prompt template txt")

    p.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    p.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base_url")

    p.add_argument("--augment-model", required=True, help="Model used for difficulty evolution")
    p.add_argument("--merged-field", default="merged_rubrics")
    p.add_argument("--response1-field", default="response_a")
    p.add_argument("--response2-field", default="response_b")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--retry", type=int, default=5)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--save-every", type=int, default=200)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    asyncio.run(
        augment_rubrics(
            input_path=Path(args.input),
            output_path=Path(args.output),
            prompt_template_path=Path(args.prompt_template),
            api_key=args.api_key,
            base_url=args.base_url,
            augment_model=args.augment_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            retry=args.retry,
            timeout_s=args.timeout,
            save_every=args.save_every,
            merged_field=args.merged_field,
            response1_field=args.response1_field,
            response2_field=args.response2_field,
        )
    )


if __name__ == "__main__":
    main()
