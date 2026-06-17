#!/usr/bin/env python3
"""
step1_build_pt_corpus.py
Build DAPT pre-training corpus.

Main experiment: pure target-domain text (no source-domain mixing)
Ablation study: add --replay to mix in-distribution training text

Usage:
  python step1_build_pt_corpus.py --data_dir ./data                  # Main: pure target domain
  python step1_build_pt_corpus.py --data_dir ./data --replay 0.006   # Ablation: 0.6% replay
"""

import json
import random
import argparse
from pathlib import Path

CENTERS = ["center1", "center2"]
SEED = 42


def load_texts(json_path: Path) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [item for item in data if len(item.get("text", "")) >= 10]


def build_corpus(data_dir: Path, center: str, replay_ratio: float = 0.0, seed: int = SEED):
    random.seed(seed)

    ood_path = data_dir / f"{center}_unlabeled.json"
    if not ood_path.exists():
        print(f"[SKIP] {ood_path} not found, run step0 first")
        return

    ood_corpus = load_texts(ood_path)

    if replay_ratio > 0:
        id_path = data_dir / "id_replay_text.json"
        if not id_path.exists():
            print(f"[WARNING] Replay file not found: {id_path}, skipping replay")
            replay_data = []
        else:
            id_corpus = load_texts(id_path)
            replay_num = max(1, int(len(ood_corpus) * replay_ratio))
            replay_data = random.sample(id_corpus, min(replay_num, len(id_corpus)))

        final_corpus = ood_corpus + replay_data
        random.shuffle(final_corpus)
        suffix = "_replay"
        print(f"[{center}] OoD: {len(ood_corpus)}, Replay: {len(replay_data)}, Total: {len(final_corpus)}")
    else:
        final_corpus = ood_corpus
        suffix = ""
        print(f"[{center}] Pure target domain: {len(final_corpus)} samples (no replay)")

    save_path = data_dir / f"{center}_pt_corpus{suffix}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(final_corpus, f, ensure_ascii=False, indent=2)
    print(f"  => {save_path}")


def update_dataset_info(data_dir: Path, replay_ratio: float = 0.0):
    """Update LLaMA-Factory dataset_info.json"""
    dataset_info_path = data_dir / "dataset_info.json"
    if dataset_info_path.exists():
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    else:
        info = {}

    suffix = "_replay" if replay_ratio > 0 else ""
    for center in CENTERS:
        key = f"{center}_pt_corpus{suffix}"
        info[key] = {
            "file_name": f"{center}_pt_corpus{suffix}.json",
            "columns": {"prompt": "text"}
        }

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"\ndataset_info.json updated: {dataset_info_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    parser.add_argument("--replay", type=float, default=0.0,
                        help="In-distribution replay ratio (0=none, 0.1=10%%). For ablation.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("===== Step 1: Build DAPT training corpus =====")
    if args.replay > 0:
        print(f"Mode: Ablation (replay_ratio={args.replay})")
    else:
        print("Mode: Main experiment (pure target domain, no replay)")
    print()

    for center in CENTERS:
        build_corpus(data_dir, center, replay_ratio=args.replay)

    update_dataset_info(data_dir, replay_ratio=args.replay)
    print("\nDone!")
