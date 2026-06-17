"""图谱概念检索模块 - 路径B"""


# 否定词列表
NEGATIONS = ["未见", "未见明显", "未见明确", "无", "正常", "显示清晰", "未显示", "显示欠清"]


def match_finding_nodes(text, kg):
    """通过别名匹配定位Finding节点，带否定过滤

    Args:
        text: 归一化后的文本
        kg: ClinicalKG实例

    Returns:
        命中的Finding实体名称列表
    """
    hits = set()

    # 遍历所有Finding类型实体
    for f_name in kg.entities_by_type("Finding"):
        attr = kg.get_entity(f_name)
        aliases = [f_name] + attr.get("aliases", [])

        # 对每个别名进行匹配
        for alias in aliases:
            idx = text.find(alias)
            if idx == -1:
                continue

            # 否定过滤：检查别名前后10字符窗口内是否有否定词
            window_start = max(0, idx - 10)
            window_end = min(len(text), idx + len(alias) + 10)
            window = text[window_start:window_end]

            # 如果窗口内有否定词，跳过此匹配
            if any(neg in window for neg in NEGATIONS):
                continue

            # 命中，记录标准实体名
            hits.add(f_name)
            break  # 找到一个别名匹配即可

    return list(hits)


def retrieve_finding_knowledge(finding_nodes, kg):
    """对命中的Finding节点，沿关系遍历取知识

    Args:
        finding_nodes: Finding实体名称列表
        kg: ClinicalKG实例

    Returns:
        知识检索结果列表，每项包含 {finding, interpretation, located_in, associated_with}
    """
    out = []

    for f in finding_nodes:
        # 1跳：获取解释
        interp = kg.get_objects(f, "indicates")

        # 1跳：获取部位
        located = kg.get_objects(f, "located_in")

        # 多跳：获取关联诊断（鉴别诊断）
        assoc = kg.get_objects(f, "associated_with")

        out.append({
            "finding": f,
            "interpretation": interp[0] if interp else "",
            "located_in": located,
            "associated_with": assoc
        })

    return out
