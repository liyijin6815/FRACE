"""Compute retrieval coverage for a KG-augmented dataset."""
import json
from collections import Counter
from .kg_loader import ClinicalKG
from .preprocessor import normalize_text
from .numeric_extractor import extract_measurements, judge_thresholds
from .graph_retriever import match_finding_nodes


def compute_coverage(input_path, kg_path):
    """Compute numeric and finding retrieval coverage.

    Args:
        input_path: Input dataset JSON path.
        kg_path: Clinical KG JSON path.

    Returns:
        Coverage counts and matched entity frequencies.
    """
    # Load the graph and source records.
    kg = ClinicalKG(kg_path)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    total_samples = len(data)
    augmented_samples = 0
    measurement_hits = Counter()
    finding_hits = Counter()
    no_hit_samples = []

    # Evaluate both retrieval paths for each record.
    for idx, sample in enumerate(data):
        text = normalize_text(sample["input"])

        # Path A: numeric measurements.
        measures = extract_measurements(text, kg)
        judgements = judge_thresholds(measures, kg)

        # Path B: qualitative findings.
        finding_nodes = match_finding_nodes(text, kg)

        # A record is covered when either path returns evidence.
        has_hit = False

        if judgements:
            has_hit = True
            for j in judgements:
                measurement_hits[j["structure"]] += 1

        if finding_nodes:
            has_hit = True
            for f in finding_nodes:
                finding_hits[f] += 1

        if has_hit:
            augmented_samples += 1
        else:
            no_hit_samples.append(idx)

    # Limit missed indices to a short diagnostic sample.
    stats = {
        "total_samples": total_samples,
        "augmented_samples": augmented_samples,
        "coverage_rate": augmented_samples / total_samples if total_samples > 0 else 0,
        "no_hit_count": len(no_hit_samples),
        "measurement_hits": dict(measurement_hits),
        "finding_hits": dict(finding_hits),
        "no_hit_sample_indices": no_hit_samples[:10]
    }

    return stats


def print_coverage_stats(stats):
    """Print coverage statistics.

    Args:
        stats: Result returned by ``compute_coverage``.
    """
    print("=" * 60)
    print("KG retrieval coverage")
    print("=" * 60)
    print(f"Total records: {stats['total_samples']}")
    print(f"Records with KG context: {stats['augmented_samples']}")
    print(f"Coverage: {stats['coverage_rate']:.2%}")
    print(f"Records without a match: {stats['no_hit_count']}")
    print()

    print("Measurement matches:")
    if stats['measurement_hits']:
        for name, count in sorted(stats['measurement_hits'].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"  {name}: {count}")
    else:
        print("  No matches")
    print()

    print("Finding matches:")
    if stats['finding_hits']:
        for name, count in sorted(stats['finding_hits'].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"  {name}: {count}")
    else:
        print("  No matches")
    print()

    if stats['no_hit_sample_indices']:
        print(f"First unmatched record indices: {stats['no_hit_sample_indices']}")
    print("=" * 60)
