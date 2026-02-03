from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import openai
from openai import AsyncOpenAI


JSONDict = Dict[str, Any]


def read_jsonl(path: Path) -> List[JSONDict]:
    rows: List[JSONDict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # best-effort: skip bad lines rather than crashing resume
                continue
    return rows


def write_jsonl(path: Path, rows: Iterable[JSONDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_template(template: str, replacements: Dict[str, str]) -> str:
    rendered = template
    for k, v in replacements.items():
        rendered = rendered.replace(k, v)
    return rendered


def ensure_record_id(row: JSONDict, index: int, id_field: Optional[str]) -> str:
    if "__record_id__" in row and str(row["__record_id__"]).strip():
        return str(row["__record_id__"])
    if id_field:
        val = row.get(id_field, "")
        if val is not None and str(val).strip():
            return str(val)
    return f"idx:{index}"


def extract_json_array(text: str) -> List[Any]:
    """
    Extract a JSON array from model output.
    Supports:
    - ```json ... ```
    - ``` ... ```
    - raw JSON
    - best-effort bracket matching
    """
    if not text:
        raise ValueError("empty model output")

    # Prefer fenced ```json blocks
    m = re.search(r"```json\\s*(\\[.*?\\])\\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))

    # Any fenced code block that looks like an array
    m = re.search(r"```\\s*(\\[.*?\\])\\s*```", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))

    # Raw JSON array
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return json.loads(stripped)

    # Best-effort: first '[' to last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        return json.loads(candidate)

    raise ValueError("could not locate JSON array in model output")


def normalize_rubric_items(items: Any) -> List[JSONDict]:
    if not isinstance(items, list):
        raise ValueError("rubrics must be a JSON array")

    normalized: List[JSONDict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        title = str(it.get("title", "")).strip()
        description = str(it.get("description", "")).strip()
        weight = it.get("weight", None)
        if description == "":
            continue
        try:
            weight_int = int(weight)
        except Exception:
            continue
        weight_int = max(0, min(10, weight_int))
        normalized.append({"title": title, "description": description, "weight": weight_int})
    return normalized


def rubrics_to_final_format(items: List[JSONDict]) -> List[JSONDict]:
    out: List[JSONDict] = []
    for it in items:
        desc = str(it.get("description", "")).strip()
        if not desc:
            continue
        points = it.get("weight", 0)
        try:
            points_int = int(points)
        except Exception:
            points_int = 0
        points_int = max(0, min(10, points_int))
        out.append({"criterion": desc, "points": points_int})
    return out


def dedup_final_rubrics(rubrics: List[JSONDict]) -> List[JSONDict]:
    """
    Deduplicate by normalized criterion text; keep the max points.
    """
    seen: Dict[str, JSONDict] = {}
    for rb in rubrics:
        crit = str(rb.get("criterion", "")).strip()
        if not crit:
            continue
        key = re.sub(r"\\s+", " ", crit).lower()
        pts = rb.get("points", 0)
        try:
            pts_int = int(pts)
        except Exception:
            pts_int = 0
        prev = seen.get(key)
        if prev is None or int(prev.get("points", 0)) < pts_int:
            seen[key] = {"criterion": crit, "points": max(0, min(10, pts_int))}
    return list(seen.values())


@dataclass
class LLMConfig:
    api_key: str
    base_url: Optional[str]
    timeout_s: int


def get_llm_config(api_key: Optional[str], base_url: Optional[str], timeout_s: int) -> LLMConfig:
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError(
            "Missing API key for this step. Provide the per-step API key "
            "(e.g., --reference-api-key / --response-api-key-a / --rubric-api-key-a / "
            "--merge-api-key / --augment-api-key), or set OPENAI_API_KEY."
        )
    return LLMConfig(api_key=key, base_url=base_url, timeout_s=timeout_s)


def make_async_client(cfg: LLMConfig) -> AsyncOpenAI:
    if cfg.base_url:
        return AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.timeout_s)
    return AsyncOpenAI(api_key=cfg.api_key, timeout=cfg.timeout_s)


async def chat_completion_with_retry(
    client: AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    retry: int,
) -> str:
    last_err: Optional[BaseException] = None
    for attempt in range(retry):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content or ""
            return content
        except (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
            openai.APIStatusError,
        ) as e:
            last_err = e
            # exponential backoff with jitter
            sleep_s = min(60.0, (2**attempt)) + random.random()
            await asyncio.sleep(sleep_s)
        except Exception as e:
            last_err = e
            break
    raise RuntimeError(str(last_err) if last_err else "unknown LLM error")


async def run_with_concurrency(
    items: List[Tuple[int, JSONDict]],
    *,
    concurrency: int,
    process_fn: Callable[[int, JSONDict], "asyncio.Future[JSONDict]"],
    save_every: int,
    save_partial_fn: Callable[[List[Optional[JSONDict]]], None],
    results: List[Optional[JSONDict]],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> None:
    if not items:
        return

    concurrency = max(1, concurrency)
    save_every = max(1, save_every)

    queue: asyncio.Queue[Optional[Tuple[int, JSONDict]]] = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)
    for _ in range(concurrency):
        queue.put_nowait(None)

    completed = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal completed
        while True:
            item = await queue.get()
            if item is None:
                return
            idx, row = item
            try:
                out = await process_fn(idx, row)
            except Exception as e:
                out = dict(row)
                out["__worker_error__"] = str(e)[:500]
            results[idx] = out

            async with lock:
                completed += 1
                if progress_cb:
                    progress_cb(completed, len(items))
                if completed % save_every == 0:
                    save_partial_fn(results)

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*workers)


def index_existing_rows(rows: List[JSONDict]) -> Dict[str, JSONDict]:
    m: Dict[str, JSONDict] = {}
    for r in rows:
        rid = r.get("__record_id__", None)
        if rid is not None and str(rid).strip():
            m[str(rid)] = r
            continue
        if "__source_index__" in r:
            m[f"idx:{r['__source_index__']}"] = r
    return m


def is_missing_or_error(row: JSONDict, field: str) -> bool:
    if row.get(f"{field}_error"):
        return True
    val = row.get(field, None)
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    if isinstance(val, list) and len(val) == 0:
        # for rubrics-like fields, empty list usually indicates failure
        return True
    return False
