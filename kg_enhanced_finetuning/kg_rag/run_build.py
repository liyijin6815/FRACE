"""入口脚本 - 运行KG增强数据集构建"""
import argparse
import os
from kg_rag.dataset_builder import build_augmented_dataset
from kg_rag.coverage_stats import compute_coverage, print_coverage_stats


def main():
    parser = argparse.ArgumentParser(
        description="构建KG增强的胎儿脑MRI诊断数据集"
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="输入JSON文件路径"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="输出JSON文件路径"
    )
    parser.add_argument(
        "--kg_path",
        default="kg_rag/clinical_kg.json",
        help="知识图谱文件路径（默认: kg_rag/clinical_kg.json）"
    )
    parser.add_argument(
        "--debug_path",
        default=None,
        help="调试信息输出路径（可选）"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="增强完成后运行覆盖率统计"
    )

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input_path):
        print(f"错误: 输入文件不存在: {args.input_path}")
        return

    if not os.path.exists(args.kg_path):
        print(f"错误: 知识图谱文件不存在: {args.kg_path}")
        return

    # 构建增强数据集
    print(f"正在处理: {args.input_path}")
    print(f"知识图谱: {args.kg_path}")
    print(f"输出路径: {args.output_path}")
    if args.debug_path:
        print(f"调试信息: {args.debug_path}")
    print()

    stats = build_augmented_dataset(
        args.input_path,
        args.output_path,
        args.kg_path,
        args.debug_path
    )

    print(f"✓ 数据集构建完成")
    print(f"  总样本数: {stats['total']}")
    print(f"  增强样本数: {stats['augmented']}")
    print(f"  覆盖率: {stats['coverage_rate']:.2%}")
    print()

    # 如果指定了--stats，运行详细统计
    if args.stats:
        print("正在计算详细统计...")
        detailed_stats = compute_coverage(args.input_path, args.kg_path)
        print_coverage_stats(detailed_stats)


if __name__ == "__main__":
    main()
