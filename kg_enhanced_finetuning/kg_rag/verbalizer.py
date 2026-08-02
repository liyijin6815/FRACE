"""语言组织模块 - 将结构化结果转换为自然语言"""


def verbalize_numeric(judgement, kg):
    """将数值判定结果组织成自然语言

    Args:
        judgement: judge_thresholds输出的单个判定结果
        kg: ClinicalKG实例

    Returns:
        自然语言句子
    """
    attr = kg.get_entity(judgement["structure"])
    template = attr.get("template", "")

    if not template:
        return ""

    # 根据方向选择合适的参数
    params = {
        "side": judgement.get("side", ""),
        "name": judgement["structure"],
        "value": round(judgement["value_cm"], 2),
        "label": judgement["label"]
    }

    # 添加正常参考值
    if judgement.get("direction") == "max":
        params["normal_max"] = judgement.get("normal_ref")
    elif judgement.get("direction") == "min":
        params["normal_min"] = judgement.get("normal_ref")

    try:
        return template.format(**params)
    except KeyError:
        # 模板参数不匹配时返回空
        return ""


def verbalize_finding(item):
    """将Finding检索结果组织成自然语言

    Args:
        item: retrieve_finding_knowledge输出的单个结果

    Returns:
        自然语言句子
    """
    s = item["interpretation"]

    # 添加鉴别诊断信息
    if item["associated_with"]:
        s += "，需与" + "、".join(item["associated_with"]) + "鉴别"

    return s
