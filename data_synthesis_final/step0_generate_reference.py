from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from common import (
    JSONDict,
    chat_completion_with_retry,
    ensure_record_id,
    get_llm_config,
    index_existing_rows,
    is_missing_or_error,
    load_text,
    make_async_client,
    read_jsonl,
    render_template,
    run_with_concurrency,
    write_jsonl,
)


async def generate_references(
    *,
    input_path: Path,
    output_path: Path,
    prompt_template_path: Path,
    question_column: str,
    id_column: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    retry: int,
    timeout_s: int,
    save_every: int,
) -> None:
    template = load_text(prompt_template_path)

    input_rows = read_jsonl(input_path)
    normalized_inputs: List[JSONDict] = []
    for idx, row in enumerate(input_rows):
        question = row.get(question_column, "")
        if question is None or str(question).strip() == "":
            continue
        out = dict(row)
        out.pop("rubric_id", None)
        out["question"] = str(question)
        if id_column and "id" not in out:
            out["id"] = row.get(id_column, "")
        out["__source_index__"] = idx
        out["__record_id__"] = ensure_record_id(out, idx, id_column)
        normalized_inputs.append(out)

    existing_map = index_existing_rows(read_jsonl(output_path))

    def _is_done(existing: JSONDict) -> bool:
        return not is_missing_or_error(existing, "reference")

    results: List[Optional[JSONDict]] = [None] * len(normalized_inputs)
    pending: List[tuple[int, JSONDict]] = []
    reused = 0
    for i, row in enumerate(normalized_inputs):
        rid = row["__record_id__"]
        exist = existing_map.get(str(rid))
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

    pbar = tqdm(total=len(pending), desc="step0: reference", ncols=120)

    async def _process(i: int, row: JSONDict) -> JSONDict:
        out = dict(row)
        out.pop("rubric_id", None)
        prompt = render_template(template, {"<<query>>": out["question"]})

        try:
            if is_missing_or_error(out, "reference"):
                out["reference"] = await chat_completion_with_retry(
                    client,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    retry=retry,
                )
                out.pop("reference_error", None)
            out["reference_model"] = model
        except Exception as e:
            out["reference"] = out.get("reference", "") or ""
            out["reference_error"] = str(e)[:500]

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

    print(f"step0 done. reused={reused} processed={len(pending)} output={output_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step0: generate reference answers (resume supported)")
    p.add_argument("--input", required=True, help="Input JSONL")
    p.add_argument("--output", required=True, help="Output JSONL")
    p.add_argument("--prompt-template", required=True, help="Prompt template txt")
    p.add_argument("--question-column", default="question", help="Column in input JSONL to use as question")
    p.add_argument("--id-column", default=None, help="Optional id column (used for resume key)")

    p.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    p.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base_url")

    p.add_argument("--model", required=True, help="Model for reference generation")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--retry", type=int, default=5)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--save-every", type=int, default=200)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    asyncio.run(
        generate_references(
            input_path=Path(args.input),
            output_path=Path(args.output),
            prompt_template_path=Path(args.prompt_template),
            question_column=args.question_column,
            id_column=args.id_column,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            retry=args.retry,
            timeout_s=args.timeout,
            save_every=args.save_every,
        )
    )


if __name__ == "__main__":
    main()
