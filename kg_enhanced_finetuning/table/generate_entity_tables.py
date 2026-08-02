"""生成知识图谱实体表格（中英文版本）

从 JSON 知识图谱中提取 entities 信息，生成结构化的 CSV 表格
包含：实体名称、类型、别名、单位、值、范围、备注等信息
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any

BASE_DIR = Path("/data/birth/lmx/work/Class_projects/lyj/work/"
                "LLM_diagnosis_evaluation/model_finetuning_v2_kg_v4/v3")

# 输入文件
KG_CN_PATH = BASE_DIR / "visualaization" / "clinical_kg.json"
KG_EN_PATH = BASE_DIR / "visualaization" / "clinical_kg_en.json"

# 输出文件
OUTPUT_CN_PATH = BASE_DIR / "table" / "entities_cn.csv"
OUTPUT_EN_PATH = BASE_DIR / "table" / "entities_en.csv"


def extract_entity_info(entity_name: str, entity_data: Dict[str, Any]) -> Dict[str, str]:
    """提取单个实体的信息，转换为表格行

    Args:
        entity_name: 实体名称
        entity_data: 实体的属性字典

    Returns:
        包含所有字段的字典
    """
    row = {
        "entity_name": entity_name,
        "type": entity_data.get("type", ""),
        "aliases": "; ".join(entity_data.get("aliases", [])),
        "unit": entity_data.get("unit", ""),
        "value": str(entity_data.get("value", "")),
        "label": entity_data.get("label", ""),
        "range": entity_data.get("range", ""),
        "has_side": str(entity_data.get("has_side", "")),
        "template": entity_data.get("template", ""),
        "note": entity_data.get("note", ""),
        "text": entity_data.get("text", ""),
    }
    return row


def generate_entity_table(kg_path: Path, output_path: Path, is_english: bool = False):
    """生成实体表格 CSV

    Args:
        kg_path: 知识图谱 JSON 文件路径
        output_path: 输出 CSV 文件路径
        is_english: 是否为英文版（影响列名）
    """
    # 读取知识图谱
    with kg_path.open("r", encoding="utf-8") as f:
        kg = json.load(f)

    entities = kg.get("entities", {})

    # 定义列名
    if is_english:
        fieldnames = [
            "Entity Name",
            "Type",
            "Aliases",
            "Unit",
            "Value",
            "Label",
            "Range",
            "Has Side",
            "Template",
            "Note",
            "Text"
        ]
        cn_to_en_fields = {
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
            "text": "Text"
        }
    else:
        fieldnames = [
            "实体名称",
            "类型",
            "别名",
            "单位",
            "值",
            "标签",
            "范围",
            "是否分侧",
            "模板",
            "备注",
            "解释文本"
        ]
        cn_to_en_fields = {
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
            "text": "解释文本"
        }

    # 提取所有实体信息
    rows = []
    for entity_name, entity_data in entities.items():
        row_data = extract_entity_info(entity_name, entity_data)
        # 转换字段名
        translated_row = {cn_to_en_fields[k]: v for k, v in row_data.items()}
        rows.append(translated_row)

    # 按类型和名称排序
    type_order = {
        "Exam": 0,
        "AnatomyDomain": 1,
        "Anatomy": 2,
        "Measurement": 3,
        "Threshold": 4,
        "Severity": 5,
        "Finding": 6,
        "Interpretation": 7,
        "Unknown": 8
    }

    type_field = "Type" if is_english else "类型"
    name_field = "Entity Name" if is_english else "实体名称"

    rows.sort(key=lambda x: (type_order.get(x[type_field], 99), x[name_field]))

    # 写入 CSV
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lang = "English" if is_english else "Chinese"
    print(f"✅ {lang} entity table → {output_path}")
    print(f"   Total entities: {len(rows)}")


def main():
    print("\n" + "="*60)
    print("知识图谱实体表格生成器")
    print("="*60)

    # 生成中文版
    print(f"\n📂 Processing Chinese version: {KG_CN_PATH}")
    generate_entity_table(KG_CN_PATH, OUTPUT_CN_PATH, is_english=False)

    # 生成英文版
    print(f"\n📂 Processing English version: {KG_EN_PATH}")
    generate_entity_table(KG_EN_PATH, OUTPUT_EN_PATH, is_english=True)

    print("\n" + "="*60)
    print("✅ 实体表格生成完成！")
    print(f"   中文版: {OUTPUT_CN_PATH}")
    print(f"   英文版: {OUTPUT_EN_PATH}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
