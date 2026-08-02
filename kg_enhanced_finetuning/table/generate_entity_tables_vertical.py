"""生成知识图谱实体表格（中英文版本） - 纵向布局

从 JSON 知识图谱中提取 entities 信息，生成纵向布局的 CSV 表格
格式：两列（属性名、属性值），每个实体占多行，实体之间用空行分隔
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path("/data/birth/lmx/work/Class_projects/lyj/work/"
                "LLM_diagnosis_evaluation/model_finetuning_v2_kg_v4/v3")

# 输入文件
KG_CN_PATH = BASE_DIR / "visualaization" / "clinical_kg.json"
KG_EN_PATH = BASE_DIR / "visualaization" / "clinical_kg_en.json"

# 输出文件
OUTPUT_CN_PATH = BASE_DIR / "table" / "entities_vertical_cn.csv"
OUTPUT_EN_PATH = BASE_DIR / "table" / "entities_vertical_en.csv"


def generate_entity_table_vertical(kg_path: Path, output_path: Path, is_english: bool = False):
    """生成实体表格 CSV（纵向布局）

    Args:
        kg_path: 知识图谱 JSON 文件路径
        output_path: 输出 CSV 文件路径
        is_english: 是否为英文版（影响字段名）
    """
    # 读取知识图谱
    with kg_path.open("r", encoding="utf-8") as f:
        kg = json.load(f)

    entities = kg.get("entities", {})

    # 定义字段名映射
    if is_english:
        field_names = {
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
        column_headers = ["Attribute", "Value"]
    else:
        field_names = {
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
        column_headers = ["属性", "值"]

    # 按类型和名称排序实体
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

    sorted_entities = sorted(
        entities.items(),
        key=lambda x: (type_order.get(x[1].get("type", "Unknown"), 99), x[0])
    )

    # 生成纵向布局的行
    rows = []

    for entity_name, entity_data in sorted_entities:
        # 实体名称（作为标题行）
        rows.append([field_names["entity_name"], entity_name])

        # 类型
        if "type" in entity_data:
            rows.append([field_names["type"], entity_data["type"]])

        # 别名
        if "aliases" in entity_data and entity_data["aliases"]:
            aliases_str = "; ".join(entity_data["aliases"])
            rows.append([field_names["aliases"], aliases_str])

        # 单位
        if "unit" in entity_data and entity_data["unit"]:
            rows.append([field_names["unit"], entity_data["unit"]])

        # 值
        if "value" in entity_data and entity_data["value"] != "":
            rows.append([field_names["value"], str(entity_data["value"])])

        # 标签
        if "label" in entity_data and entity_data["label"]:
            rows.append([field_names["label"], entity_data["label"]])

        # 范围
        if "range" in entity_data and entity_data["range"]:
            rows.append([field_names["range"], entity_data["range"]])

        # 是否分侧
        if "has_side" in entity_data and entity_data["has_side"] != "":
            rows.append([field_names["has_side"], str(entity_data["has_side"])])

        # 模板
        if "template" in entity_data and entity_data["template"]:
            rows.append([field_names["template"], entity_data["template"]])

        # 备注
        if "note" in entity_data and entity_data["note"]:
            rows.append([field_names["note"], entity_data["note"]])

        # 解释文本
        if "text" in entity_data and entity_data["text"]:
            rows.append([field_names["text"], entity_data["text"]])

        # 实体之间添加空行分隔
        rows.append(["", ""])

    # 写入 CSV
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(column_headers)
        # 写入所有行
        writer.writerows(rows)

    lang = "English" if is_english else "Chinese"
    print(f"✅ {lang} entity table (vertical layout) → {output_path}")
    print(f"   Total entities: {len(sorted_entities)}")
    print(f"   Total rows: {len(rows)}")


def main():
    print("\n" + "="*60)
    print("知识图谱实体表格生成器（纵向布局）")
    print("="*60)

    # 生成中文版
    print(f"\n📂 Processing Chinese version: {KG_CN_PATH}")
    generate_entity_table_vertical(KG_CN_PATH, OUTPUT_CN_PATH, is_english=False)

    # 生成英文版
    print(f"\n📂 Processing English version: {KG_EN_PATH}")
    generate_entity_table_vertical(KG_EN_PATH, OUTPUT_EN_PATH, is_english=True)

    print("\n" + "="*60)
    print("✅ 实体表格生成完成！")
    print(f"   中文版: {OUTPUT_CN_PATH}")
    print(f"   英文版: {OUTPUT_EN_PATH}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
