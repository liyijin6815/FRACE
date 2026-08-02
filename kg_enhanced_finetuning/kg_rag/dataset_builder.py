"""数据集构建模块 - 将KG上下文拼接到输入末尾"""
import json
from .kg_loader import ClinicalKG
from .context_builder import build_kg_context


# 指令后缀：告知模型如何使用临床知识参考
INSTR_SUFFIX = "如果输入中包含【临床知识参考】，请将其作为辅助判断依据，" \
               "但诊断意见仍以影像所见为准，不要机械复述参考内容。"


def build_augmented_dataset(input_path, output_path, kg_path, debug_path=None):
    """构建KG增强的数据集

    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        kg_path: 知识图谱文件路径
        debug_path: 可选，调试信息输出路径

    Returns:
        统计信息字典 {total, augmented, coverage_rate}
    """
    # 加载知识图谱
    kg = ClinicalKG(kg_path)

    # 加载数据集
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    debug = []
    augmented_count = 0

    # 处理每个样本
    for sample in data:
        # 构建KG上下文
        ctx = build_kg_context(sample["input"], kg)

        # 如果有KG上下文，拼接到input末尾并修改instruction
        if ctx:
            sample["input"] = sample["input"] + "\n\n" + ctx
            sample["instruction"] = sample["instruction"].rstrip() + INSTR_SUFFIX
            augmented_count += 1

        # output字段绝不修改

        # 记录调试信息
        if debug_path is not None:
            debug.append({
                "input": sample["input"],
                "kg_context": ctx,
                "has_kg": bool(ctx)
            })

    # 保存增强后的数据集
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 保存调试信息
    if debug_path:
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)

    # 返回统计信息
    stats = {
        "total": len(data),
        "augmented": augmented_count,
        "coverage_rate": augmented_count / len(data) if data else 0
    }

    return stats
