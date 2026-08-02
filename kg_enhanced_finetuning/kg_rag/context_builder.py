"""上下文构建模块 - 顶层编排，调用路径A+B并合并"""
from .preprocessor import normalize_text
from .numeric_extractor import extract_measurements, judge_thresholds
from .graph_retriever import match_finding_nodes, retrieve_finding_knowledge
from .verbalizer import verbalize_numeric, verbalize_finding


def build_kg_context(report_text, kg):
    """为单个报告构建KG增强的临床知识参考文本

    Args:
        report_text: 报告所见文本
        kg: ClinicalKG实例

    Returns:
        【临床知识参考】文本，无命中则返回空字符串
    """
    # 文本预处理
    text = normalize_text(report_text)

    lines = []

    # 路径A：数值规则化提取
    measures = extract_measurements(text, kg)
    judgements = judge_thresholds(measures, kg)
    for j in judgements:
        line = verbalize_numeric(j, kg)
        if line:
            lines.append(line)

    # 路径B：图谱概念检索
    finding_nodes = match_finding_nodes(text, kg)
    finding_knowledge = retrieve_finding_knowledge(finding_nodes, kg)
    for item in finding_knowledge:
        line = verbalize_finding(item)
        if line:
            lines.append(line)

    # 合并去重，保持顺序
    seen = set()
    dedup = []
    for ln in lines:
        if ln and ln not in seen:
            seen.add(ln)
            dedup.append(ln)

    # 无命中返回空字符串
    if not dedup:
        return ""

    # 组织成【临床知识参考】格式
    return "【临床知识参考】\n" + "\n".join("- " + x for x in dedup)
