#!/usr/bin/env python3
"""Generate horizontal and vertical CSV entity tables from a KG JSON file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "entity_name",
    "type",
    "aliases",
    "unit",
    "value",
    "label",
    "range",
    "has_side",
    "template",
    "note",
    "text",
)

HEADERS = {
    "en": {
        "entity_name": "Entity Name",
        "type": "Type",
        "aliases": "Aliases",
        "unit": "Unit",
        "value": "Value",
        "label": "Label",
        "range": "Range",
        "has_side": "Has Side",
        "template": "Template",
        "note": "Note",
        "text": "Text",
    },
    "cn": {
        "entity_name": "实体名称",
        "type": "类型",
        "aliases": "别名",
        "unit": "单位",
        "value": "值",
        "label": "标签",
        "range": "范围",
        "has_side": "是否分侧",
        "template": "模板",
        "note": "备注",
        "text": "解释文本",
    },
}

TYPE_ORDER = {
    "Exam": 0,
    "AnatomyDomain": 1,
    "Anatomy": 2,
    "Measurement": 3,
    "Threshold": 4,
    "Severity": 5,
    "Finding": 6,
    "Interpretation": 7,
    "Unknown": 8,
}


def entity_rows(kg_path: Path) -> list[dict[str, str]]:
    with kg_path.open("r", encoding="utf-8") as handle:
        graph = json.load(handle)
    entities = graph.get("entities", {})
    rows = []
    for name, attributes in entities.items():
        rows.append(
            {
                "entity_name": name,
                "type": str(attributes.get("type", "")),
                "aliases": "; ".join(attributes.get("aliases", [])),
                "unit": str(attributes.get("unit", "")),
                "value": str(attributes.get("value", "")),
                "label": str(attributes.get("label", "")),
                "range": str(attributes.get("range", "")),
                "has_side": str(attributes.get("has_side", "")),
                "template": str(attributes.get("template", "")),
                "note": str(attributes.get("note", "")),
                "text": str(attributes.get("text", "")),
            }
        )
    rows.sort(key=lambda row: (TYPE_ORDER.get(row["type"], 99), row["entity_name"]))
    return rows


def write_horizontal(rows: list[dict[str, str]], path: Path, language: str) -> None:
    headers = HEADERS[language]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[headers[field] for field in FIELDS])
        writer.writeheader()
        for row in rows:
            writer.writerow({headers[field]: row[field] for field in FIELDS})


def write_vertical(rows: list[dict[str, str]], path: Path, language: str) -> None:
    headers = HEADERS[language]
    column_headers = ("Attribute", "Value") if language == "en" else ("属性", "值")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(column_headers)
        for row in rows:
            for field in FIELDS:
                if row[field] != "":
                    writer.writerow((headers[field], row[field]))
            writer.writerow(("", ""))


def parse_args() -> argparse.Namespace:
    table_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg-path", type=Path, required=True)
    parser.add_argument("--language", choices=("cn", "en"), required=True)
    parser.add_argument("--output-dir", type=Path, default=table_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.kg_path.is_file():
        raise FileNotFoundError(f"Knowledge graph does not exist: {args.kg_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = entity_rows(args.kg_path)
    horizontal = args.output_dir / f"entities_{args.language}.csv"
    vertical = args.output_dir / f"entities_vertical_{args.language}.csv"
    write_horizontal(rows, horizontal, args.language)
    write_vertical(rows, vertical, args.language)
    print(f"Generated {horizontal} and {vertical} from {len(rows)} entities")


if __name__ == "__main__":
    main()
