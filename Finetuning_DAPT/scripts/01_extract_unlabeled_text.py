#!/usr/bin/env python3
"""Extract unlabeled target-domain text for DAPT.

Only the ``input`` field is used. The diagnostic ``output`` field is never
included in the continued-pretraining corpus.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


OOD_FILES = {
    "center1": (
        "OutofDistribution_center1_normal.json",
        "OutofDistribution_center1_patients.json",
    ),
    "center2": (
        "OutofDistribution_center2_normal.json",
        "OutofDistribution_center2_patients.json",
    ),
}

# These expressions remove common administrative fields while retaining
# clinical findings. The Chinese terms are functional matching patterns.
ADMINISTRATIVE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"检查号[:：].*",
        r"报告医师[:：].*",
        r"审核医师[:：].*",
        r"医院名称[:：].*",
        r"检查时间[:：].*",
        r"患者姓名[:：].*",
        r"申请科室[:：].*",
    )
)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove administrative lines."""
    text = re.sub(r"[ \t]+", " ", str(text).strip())
    text = re.sub(r"\n+", "\n", text)
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(pattern.match(line) for pattern in ADMINISTRATIVE_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def extract_input_only(path: Path) -> list[dict[str, str]]:
    """Return valid ``input`` values as LLaMA-Factory PT records."""
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list: {path}")

    extracted = []
    for record in records:
        if not isinstance(record, dict):
            continue
        text = clean_text(record.get("input", ""))
        if len(text) >= 10:
            extracted.append({"text": text})
    return extracted


def write_json(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-train-file",
        default="InDistribution_train.json",
        help="Source-domain file used only to construct the replay pool.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for center, filenames in OOD_FILES.items():
        center_records = []
        for filename in filenames:
            path = args.dataset_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing target-domain dataset: {path}")
            records = extract_input_only(path)
            print(f"{filename}: {len(records)} unlabeled records")
            center_records.extend(records)
        write_json(center_records, args.output_dir / f"{center}_unlabeled.json")

    source_path = args.dataset_dir / args.source_train_file
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source-domain training dataset: {source_path}")
    write_json(extract_input_only(source_path), args.output_dir / "id_replay_text.json")


if __name__ == "__main__":
    main()
