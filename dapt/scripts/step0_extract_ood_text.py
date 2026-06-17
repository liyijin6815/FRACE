#!/usr/bin/env python3
"""
step0_extract_ood_text.py
Extract unlabeled text from OoD datasets for DAPT continual pre-training.

Only extracts the `input` field (imaging findings), NOT the `output` field 
(diagnosis/labels), to maintain the unsupervised domain adaptation setting.

Usage:
  python step0_extract_ood_text.py --dataset_dir /path/to/dataset --output_dir ./data
"""

import json
import re
import argparse
from pathlib import Path

CENTERS = ["center1", "center2"]
OOD_DATASETS = {
    "center1": [
        "OutofDistribution_center1_normal.json",
        "OutofDistribution_center1_patients.json",
    ],
    "center2": [
        "OutofDistribution_center2_normal.json",
        "OutofDistribution_center2_patients.json",
    ],
}


def clean_text(text: str) -> str:
    """Clean report text: remove administrative info, keep medical content."""
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    drop_patterns = [
        r"检查号[:：].*",
        r"报告医师[:：].*",
        r"审核医师[:：].*",
        r"医院名称[:：].*",
        r"检查时间[:：].*",
        r"患者姓名[:：].*",
        r"申请科室[:：].*",
    ]
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(re.match(p, line) for p in drop_patterns):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def extract_input_only(filepath: Path) -> list[dict]:
    """Extract only input field (imaging findings) as unlabeled text."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    for item in data:
        text = item.get("input", "").strip()
        if text:
            text = clean_text(text)
            if len(text) >= 10:
                texts.append({"text": text})
    return texts


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(data)} samples -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Path to raw dataset directory")
    parser.add_argument("--output_dir", default="./data", help="Output directory")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== Step 0: Extract unlabeled text (input field only) =====\n")

    for center, files in OOD_DATASETS.items():
        texts = []
        for fname in files:
            fpath = dataset_dir / fname
            if not fpath.exists():
                print(f"  [WARNING] File not found: {fpath}")
                continue
            file_texts = extract_input_only(fpath)
            print(f"  {fname}: {len(file_texts)} samples")
            texts.extend(file_texts)
        save_json(texts, output_dir / f"{center}_unlabeled.json")
        print()

    # Extract in-distribution training text for replay ablation
    print("Extracting in-distribution training text (for replay ablation)...")
    id_train_path = dataset_dir / "InDistribution_train.json"
    if id_train_path.exists():
        id_texts = extract_input_only(id_train_path)
        save_json(id_texts, output_dir / "id_replay_text.json")
    else:
        print(f"  [WARNING] ID training set not found: {id_train_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
