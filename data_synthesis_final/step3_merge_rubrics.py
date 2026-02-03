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


async def merge_rubrics(
    *,
    input_path: Path,
    output_path: Path,
    prompt_template_path: Path,
    api_key: Optional[str],
    base_url: Optional[str],
    merge_model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    retry: int,
    timeout_s: int,
    save_every: int,
    rubrics_a_field: str,
    rubrics_b_field: str,
) -> None:
    template = load_text(prompt_template_path)
    input_rows = read_jsonl(input_path)
    existing_map = index_existing_rows(read_jsonl(output_path))

    def _is_done(existing: JSONDict) -> bool:
        return not is_missing_or_error(existing, "merged_rubrics")

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

    pbar = tqdm(total=len(pending), desc="step3: merge", ncols=120)

    async def _process(i: int, row: JSONDict) -> JSONDict:
        out = dict(row)
        out.pop("rubric_id", None)
        q = str(out.get("question", "")).strip()
        rub_a = out.get(rubrics_a_field, [])
        rub_b = out.get(rubrics_b_field, [])

        if not isinstance(rub_a, list) or len(rub_a) == 0:
            out["merged_rubrics"] = []
            out["merged_rubrics_error"] = f"missing {rubrics_a_field}"
            return out

        if not isinstance(rub_b, list) or len(rub_b) == 0:
            # no second set -> passthrough
            out["merged_rubrics"] = rub_a
            out.pop("merged_rubrics_error", None)
            out["merged_rubrics_model"] = "passthrough"
            return out

        prompt = render_template(
            template,
            {
                "{|prompt|}": q,
                "{|rubrics1|}": json.dumps(rub_a, ensure_ascii=False),
                "{|rubrics2|}": json.dumps(rub_b, ensure_ascii=False),
            },
        )

        try:
            if is_missing_or_error(out, "merged_rubrics"):
                text = await chat_completion_with_retry(
                    client,
                    model=merge_model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    retry=retry,
                )
                arr = extract_json_array(text)
                out["merged_rubrics"] = normalize_rubric_items(arr)
                out.pop("merged_rubrics_error", None)
            out["merged_rubrics_model"] = merge_model
        except Exception as e:
            out["merged_rubrics"] = out.get("merged_rubrics", []) or []
            out["merged_rubrics_error"] = str(e)[:500]

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

    print(f"step3 done. reused={reused} processed={len(pending)} output={output_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step3: merge rubrics (multi-model aggregation)")
    p.add_argument("--input", required=True, help="Input JSONL (from step1)")
    p.add_argument("--output", required=True, help="Output JSONL")
    p.add_argument("--prompt-template", required=True, help="Merge prompt template txt")

    p.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    p.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base_url")

    p.add_argument("--merge-model", required=True, help="Model used to merge rubrics")
    p.add_argument("--rubrics-a-field", default="rubrics_a")
    p.add_argument("--rubrics-b-field", default="rubrics_b")
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
        merge_rubrics(
            input_path=Path(args.input),
            output_path=Path(args.output),
            prompt_template_path=Path(args.prompt_template),
            api_key=args.api_key,
            base_url=args.base_url,
            merge_model=args.merge_model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            retry=args.retry,
            timeout_s=args.timeout,
            save_every=args.save_every,
            rubrics_a_field=args.rubrics_a_field,
            rubrics_b_field=args.rubrics_b_field,
        )
    )


if __name__ == "__main__":
    main()
