"""Command-line entry point for KG-augmented dataset construction."""

import argparse
from pathlib import Path

from .coverage_stats import compute_coverage, print_coverage_stats
from .dataset_builder import build_augmented_dataset


DEFAULT_KG_PATH = Path(__file__).with_name("clinical_kg.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--kg-path", type=Path, default=DEFAULT_KG_PATH)
    parser.add_argument("--debug-path", type=Path)
    parser.add_argument("--stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {args.input_path}")
    if not args.kg_path.is_file():
        raise FileNotFoundError(f"Knowledge graph does not exist: {args.kg_path}")

    print(f"Input dataset: {args.input_path}")
    print(f"Knowledge graph: {args.kg_path}")
    print(f"Output dataset: {args.output_path}")
    if args.debug_path:
        print(f"Debug output: {args.debug_path}")

    stats = build_augmented_dataset(
        args.input_path,
        args.output_path,
        args.kg_path,
        args.debug_path,
    )
    print(f"Processed records: {stats['total']}")
    print(f"Records with KG context: {stats['augmented']}")
    print(f"Coverage: {stats['coverage_rate']:.2%}")

    if args.stats:
        print_coverage_stats(compute_coverage(args.input_path, args.kg_path))


if __name__ == "__main__":
    main()
