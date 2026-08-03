"""Append retrieved KG context to model inputs."""

import json
from pathlib import Path

from .context_builder import build_kg_context
from .kg_loader import ClinicalKG


# This Chinese instruction is model-facing content, not a code comment.
INSTRUCTION_SUFFIX = (
    "如果输入中包含【临床知识参考】，请将其作为辅助判断依据，"
    "但诊断意见仍以影像所见为准，不要机械复述参考内容。"
)


def build_augmented_dataset(input_path, output_path, kg_path, debug_path=None):
    """Build one KG-augmented dataset.

    Args:
        input_path: Source JSON list with ``instruction`` and ``input`` fields.
        output_path: Destination JSON path.
        kg_path: Clinical KG JSON path.
        debug_path: Optional path for per-record retrieval diagnostics.

    Returns:
        A dictionary containing total records, augmented records, and coverage.
    """
    kg = ClinicalKG(kg_path)
    with open(input_path, encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list: {input_path}")

    diagnostics = []
    augmented_count = 0
    augmented_records = []

    for index, source_record in enumerate(records):
        if not isinstance(source_record, dict):
            raise ValueError(f"Record {index} is not a JSON object")
        if "input" not in source_record or "instruction" not in source_record:
            raise KeyError(f"Record {index} lacks instruction or input")

        record = dict(source_record)
        context = build_kg_context(record["input"], kg)
        if context:
            record["input"] = f"{record['input']}\n\n{context}"
            record["instruction"] = record["instruction"].rstrip() + INSTRUCTION_SUFFIX
            augmented_count += 1
        augmented_records.append(record)

        if debug_path is not None:
            diagnostics.append(
                {
                    "record_index": index,
                    "kg_context": context,
                    "has_kg": bool(context),
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(augmented_records, handle, ensure_ascii=False, indent=2)

    if debug_path is not None:
        debug_path = Path(debug_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("w", encoding="utf-8") as handle:
            json.dump(diagnostics, handle, ensure_ascii=False, indent=2)

    total = len(augmented_records)
    return {
        "total": total,
        "augmented": augmented_count,
        "coverage_rate": augmented_count / total if total else 0.0,
    }
