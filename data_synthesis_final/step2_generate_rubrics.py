from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm
from openai import AsyncOpenAI

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


async def generate_rubrics(
    *,
    input_path: Path,
    output_path: Path,
    prompt_template_path: Path,
    api_key: Optional[str],
    base_url: Optional[str],
    api_key_a: Optional[str] = None,
    base_url_a: Optional[str] = None,
    api_key_b: Optional[str] = None,
    base_url_b: Optional[str] = None,
    model_a: str,
    model_b: Optional[str],
    max_tokens: int,
    temperature: float,
    concurrency: int,
    retry: int,
    timeout_s: int,
    save_every: int,
    reference_field: str,
) -> None:
    template = load_text(prompt_template_path)
    input_rows = read_jsonl(input_path)
    existing_map = index_existing_rows(read_jsonl(output_path))

    def _is_done(existing: JSONDict) -> bool:
        if is_missing_or_error(existing, "rubrics_a"):
            return False
        if model_b and is_missing_or_error(existing, "rubrics_b"):
            return False
        return True

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

    def _pick(primary: Optional[str], fallback: Optional[str]) -> Optional[str]:
        if primary is None:
            return fallback
        if isinstance(primary, str) and primary.strip() == "":
            return fallback
        return primary

    cfg_a = get_llm_config(_pick(api_key_a, api_key), _pick(base_url_a, base_url), timeout_s)
    client_a = make_async_client(cfg_a)

    client_b = None
    if model_b:
        cfg_b = get_llm_config(_pick(api_key_b, api_key), _pick(base_url_b, base_url), timeout_s)
        client_b = client_a if cfg_b == cfg_a else make_async_client(cfg_b)

    pbar = tqdm(total=len(pending), desc="step2: rubrics", ncols=120)

    async def _gen_one(client: AsyncOpenAI, model: str, prompt: str) -> List[JSONDict]:
        text = await chat_completion_with_retry(
            client,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            retry=retry,
        )
        arr = extract_json_array(text)
        return normalize_rubric_items(arr)

    async def _process(i: int, row: JSONDict) -> JSONDict:
        out = dict(row)
        out.pop("rubric_id", None)
        q = str(out.get("question", "")).strip()
        if not q:
            out["rubrics_a"] = []
            out["rubrics_a_error"] = "missing question"
            return out

        ref = str(out.get(reference_field, "")).strip()
        if not ref:
            out["rubrics_a"] = []
            out["rubrics_a_error"] = f"missing {reference_field}"
            return out

        prompt = render_template(template, {"<<query>>": q, "<<reference>>": ref})
        try:
            if is_missing_or_error(out, "rubrics_a"):
                out["rubrics_a"] = await _gen_one(client_a, model_a, prompt)
                out.pop("rubrics_a_error", None)
            out["rubrics_a_model"] = model_a
        except Exception as e:
            out["rubrics_a"] = out.get("rubrics_a", []) or []
            out["rubrics_a_error"] = str(e)[:500]

        if model_b:
            try:
                if is_missing_or_error(out, "rubrics_b"):
                    if client_b is None:
                        raise RuntimeError("missing client_b for model_b")
                    out["rubrics_b"] = await _gen_one(client_b, model_b, prompt)
                    out.pop("rubrics_b_error", None)
                out["rubrics_b_model"] = model_b
            except Exception as e:
                out["rubrics_b"] = out.get("rubrics_b", []) or []
                out["rubrics_b_error"] = str(e)[:500]

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
    await client_a.close()
    if client_b is not None and client_b is not client_a:
        await client_b.close()

    print(f"step2 done. reused={reused} processed={len(pending)} output={output_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step2: generate candidate rubrics (resume supported)")
    p.add_argument("--input", required=True, help="Input JSONL (from step0)")
    p.add_argument("--output", required=True, help="Output JSONL")
    p.add_argument("--prompt-template", required=True, help="Prompt template txt")

    p.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    p.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base_url")
    p.add_argument("--api-key-a", default=None, help="Optional API key for model A (fallback to --api-key/env)")
    p.add_argument("--base-url-a", default=None, help="Optional base_url for model A (fallback to --base-url)")
    p.add_argument("--api-key-b", default=None, help="Optional API key for model B (fallback to --api-key/env)")
    p.add_argument("--base-url-b", default=None, help="Optional base_url for model B (fallback to --base-url)")

    p.add_argument("--model-a", required=True, help="Model for rubric A")
    p.add_argument("--model-b", default=None, help="Optional model for rubric B")
    p.add_argument("--reference-field", default="reference")
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
        generate_rubrics(
            input_path=Path(args.input),
            output_path=Path(args.output),
            prompt_template_path=Path(args.prompt_template),
            api_key=args.api_key,
            base_url=args.base_url,
            api_key_a=args.api_key_a,
            base_url_a=args.base_url_a,
            api_key_b=args.api_key_b,
            base_url_b=args.base_url_b,
            model_a=args.model_a,
            model_b=args.model_b,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            retry=args.retry,
            timeout_s=args.timeout,
            save_every=args.save_every,
            reference_field=args.reference_field,
        )
    )


if __name__ == "__main__":
    main()
