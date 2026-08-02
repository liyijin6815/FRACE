"""覆盖率统计模块"""
import json
from collections import Counter
from .kg_loader import ClinicalKG
from .preprocessor import normalize_text
from .numeric_extractor import extract_measurements, judge_thresholds
from .graph_retriever import match_finding_nodes


def compute_coverage(input_path, kg_path):
    """计算KG检索的覆盖率统计

    Args:
        input_path: 输入JSON文件路径
        kg_path: 知识图谱文件路径

    Returns:
        统计信息字典
    """
    # 加载知识图谱和数据
    kg = ClinicalKG(kg_path)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    total_samples = len(data)
    augmented_samples = 0
    measurement_hits = Counter()
    finding_hits = Counter()
    no_hit_samples = []

    # 统计每个样本
    for idx, sample in enumerate(data):
        text = normalize_text(sample["input"])

        # 路径A：数值提取
        measures = extract_measurements(text, kg)
        judgements = judge_thresholds(measures, kg)

        # 路径B：Finding匹配
        finding_nodes = match_finding_nodes(text, kg)

        # 记录命中
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

    # 构建统计结果
    stats = {
        "total_samples": total_samples,
        "augmented_samples": augmented_samples,
        "coverage_rate": augmented_samples / total_samples if total_samples > 0 else 0,
        "no_hit_count": len(no_hit_samples),
        "measurement_hits": dict(measurement_hits),
        "finding_hits": dict(finding_hits),
        "no_hit_sample_indices": no_hit_samples[:10]  # 只显示前10个
    }

    return stats


def print_coverage_stats(stats):
    """打印覆盖率统计信息

    Args:
        stats: compute_coverage返回的统计字典
    """
    print("=" * 60)
    print("KG检索覆盖率统计")
    print("=" * 60)
    print(f"总样本数: {stats['total_samples']}")
    print(f"生成参考的样本数: {stats['augmented_samples']}")
    print(f"覆盖率: {stats['coverage_rate']:.2%}")
    print(f"未命中任何规则的样本数: {stats['no_hit_count']}")
    print()

    print("Measurement命中统计:")
    if stats['measurement_hits']:
        for name, count in sorted(stats['measurement_hits'].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"  {name}: {count}次")
    else:
        print("  无命中")
    print()

    print("Finding命中统计:")
    if stats['finding_hits']:
        for name, count in sorted(stats['finding_hits'].items(),
                                   key=lambda x: x[1], reverse=True):
            print(f"  {name}: {count}次")
    else:
        print("  无命中")
    print()

    if stats['no_hit_sample_indices']:
        print(f"未命中样本索引（前10个）: {stats['no_hit_sample_indices']}")
    print("=" * 60)
