from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from common import (
    JSONDict,
    dedup_final_rubrics,
    read_jsonl,
    rubrics_to_final_format,
    write_jsonl,
)


def write_parquet(path: Path, rows: List[JSONDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # Define nested schema: rubrics = list<struct<criterion: string, points: int32>>
    rubric_struct = pa.struct([("criterion", pa.string()), ("points", pa.int32())])
    schema = pa.schema(
        [
            ("question", pa.string()),
            ("id", pa.string()),
            ("rubrics", pa.list_(rubric_struct)),
        ]
    )

    questions: List[str] = []
    ids: List[str] = []
    rubrics_col: List[Any] = []

    for r in rows:
        questions.append(str(r.get("question", "")))
        ids.append(str(r.get("id", "")))
        rubrics_col.append(r.get("rubrics", []))

    table = pa.Table.from_arrays(
        [
            pa.array(questions, type=pa.string()),
            pa.array(ids, type=pa.string()),
            pa.array(rubrics_col, type=pa.list_(rubric_struct)),
        ],
        schema=schema,
    )
    pq.write_table(table, path)


def export_dataset(
    *,
    input_path: Path,
    output_jsonl: Path,
    output_parquet: Path,
    base_field: str,
    augment_field: str,
    max_criteria: Optional[int],
    skip_missing: bool,
) -> None:
    rows = read_jsonl(input_path)

    final_rows: List[JSONDict] = []
    for row in rows:
        q = str(row.get("question", "")).strip()
        if not q:
            continue

        base = row.get(base_field, [])
        aug = row.get(augment_field, [])
        if not isinstance(base, list):
            base = []
        if not isinstance(aug, list):
            aug = []

        if skip_missing and len(base) == 0:
            continue

        combined = base + aug
        final_rubrics = rubrics_to_final_format(combined)
        final_rubrics = dedup_final_rubrics(final_rubrics)
        if max_criteria is not None and max_criteria > 0:
            final_rubrics = final_rubrics[: max_criteria]

        if skip_missing and len(final_rubrics) == 0:
            continue

        final_rows.append(
            {
                "question": q,
                "id": str(row.get("id", "")),
                "rubrics": final_rubrics,
            }
        )

    write_jsonl(output_jsonl, final_rows)
    write_parquet(output_parquet, final_rows)
    print(f"exported jsonl={output_jsonl} parquet={output_parquet} rows={len(final_rows)}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Step5: export final dataset to JSONL + Parquet")
    p.add_argument("--input", required=True, help="Input JSONL (from step3)")
    p.add_argument("--output-jsonl", required=True, help="Final output JSONL (viewable)")
    p.add_argument("--output-parquet", required=True, help="Final output Parquet (primary)")
    p.add_argument("--base-field", default="merged_rubrics")
    p.add_argument("--augment-field", default="augmented_rubrics")
    p.add_argument("--max-criteria", type=int, default=None)
    p.add_argument("--skip-missing", action="store_true", help="Skip rows missing base rubrics")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    export_dataset(
        input_path=Path(args.input),
        output_jsonl=Path(args.output_jsonl),
        output_parquet=Path(args.output_parquet),
        base_field=args.base_field,
        augment_field=args.augment_field,
        max_criteria=args.max_criteria,
        skip_missing=args.skip_missing,
    )


if __name__ == "__main__":
    main()
