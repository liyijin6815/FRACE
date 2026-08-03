#!/usr/bin/env python3
"""Build center-specific DAPT corpora with optional source replay."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


CENTERS = ("center1", "center2")


def load_texts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return [
        {"text": str(item["text"]).strip()}
        for item in records
        if isinstance(item, dict) and len(str(item.get("text", "")).strip()) >= 10
    ]


def build_corpus(data_dir: Path, center: str, replay_ratio: float, seed: int) -> Path:
    target_path = data_dir / f"{center}_unlabeled.json"
    source_path = data_dir / "id_replay_text.json"
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing target corpus: {target_path}")

    target_records = load_texts(target_path)
    replay_records = []
    if replay_ratio > 0:
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source replay pool: {source_path}")
        source_records = load_texts(source_path)
        replay_count = min(
            len(source_records), max(1, int(len(target_records) * replay_ratio))
        )
        replay_records = random.Random(seed).sample(source_records, replay_count)

    combined = target_records + replay_records
    random.Random(seed).shuffle(combined)
    suffix = "_replay" if replay_ratio > 0 else ""
    output_path = data_dir / f"{center}_pt_corpus{suffix}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, ensure_ascii=False, indent=2)

    print(
        f"{center}: target={len(target_records)}, replay={len(replay_records)}, "
        f"total={len(combined)} -> {output_path}"
    )
    return output_path


def update_dataset_info(data_dir: Path, replay_enabled: bool) -> None:
    suffix = "_replay" if replay_enabled else ""
    registry = {
        f"{center}_pt_corpus{suffix}": {
            "file_name": f"{center}_pt_corpus{suffix}.json",
            "columns": {"prompt": "text"},
        }
        for center in CENTERS
    }
    output_path = data_dir / "dataset_info.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(registry, handle, ensure_ascii=False, indent=2)
    print(f"Updated dataset registry: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--replay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.replay < 0:
        parser.error("--replay must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for center in CENTERS:
        build_corpus(args.data_dir, center, args.replay, args.seed)
    update_dataset_info(args.data_dir, replay_enabled=args.replay > 0)


if __name__ == "__main__":
    main()
