"""数值规则化提取模块 - 路径A"""
import re
from .preprocessor import split_sentences


def to_cm(value, unit):
    """单位统一转换为cm

    Args:
        value: 数值
        unit: 单位（cm或mm）

    Returns:
        转换为cm的数值
    """
    if unit.lower() == "mm":
        return value / 10.0
    return value


def _match_value_with_side(sentence, alias):
    """从句子中提取数值和侧别信息

    Args:
        sentence: 句子文本
        alias: 结构别名

    Yields:
        (side, value, unit) 元组
    """
    # 侧别关键词
    side_patterns = [
        (r'右侧', '右侧'),
        (r'左侧', '左侧'),
        (r'双侧', '双侧'),
    ]

    # 数值模式：支持多种表述方式
    # 宽约1.07cm / 约10.7mm / 测值1.07cm / 宽径约1.07cm / 深约1.0cm
    value_patterns = [
        r'约\s*(\d+\.?\d*)\s*(cm|mm)',
        r'宽约\s*(\d+\.?\d*)\s*(cm|mm)',
        r'深约\s*(\d+\.?\d*)\s*(cm|mm)',
        r'测值\s*(\d+\.?\d*)\s*(cm|mm)',
        r'宽径约\s*(\d+\.?\d*)\s*(cm|mm)',
        r'为\s*(\d+\.?\d*)\s*(cm|mm)',
        r'达\s*(\d+\.?\d*)\s*(cm|mm)',
        r'[\s:](\d+\.?\d*)\s*(cm|mm)',
    ]

    # 检查是否包含别名
    if alias not in sentence:
        return

    # 提取侧别
    side = ""
    for pattern, side_name in side_patterns:
        if re.search(pattern, sentence):
            side = side_name
            break

    # 提取数值
    for pattern in value_patterns:
        matches = re.finditer(pattern, sentence, re.IGNORECASE)
        for match in matches:
            value = float(match.group(1))
            unit = match.group(2).lower()
            yield (side, value, unit)


def extract_measurements(text, kg):
    """从文本中提取测量值

    Args:
        text: 归一化后的文本
        kg: ClinicalKG实例

    Returns:
        测量结果列表，每项包含 {structure, side, value_cm, raw_sentence}
    """
    results = []
    sentences = split_sentences(text)

    # 遍历所有Measurement类型实体
    for m_name in kg.entities_by_type("Measurement"):
        attr = kg.get_entity(m_name)
        aliases = [m_name] + attr.get("aliases", [])

        # 对每个别名在每个句子中匹配
        for alias in aliases:
            for sent in sentences:
                for side, value, unit in _match_value_with_side(sent, alias):
                    results.append({
                        "structure": m_name,
                        "side": side,
                        "value_cm": to_cm(value, unit),
                        "raw_sentence": sent
                    })

    return results


def _parse_threshold_value(threshold_str):
    """从阈值实体名称中提取数值

    Args:
        threshold_str: 阈值字符串，如 "阈值_1.0cm" 或 "1.0"

    Returns:
        浮点数值
    """
    import re
    # 尝试匹配 "阈值_数值cm" 格式
    match = re.search(r'(\d+\.?\d*)', threshold_str)
    if match:
        return float(match.group(1))
    # 如果已经是纯数字，直接转换
    return float(threshold_str)


def judge_thresholds(measurements, kg):
    """根据阈值判断测量值的严重程度

    Args:
        measurements: extract_measurements的输出
        kg: ClinicalKG实例

    Returns:
        判定结果列表，每项包含测量信息+分级标签
    """
    out = []

    for m in measurements:
        s = m["structure"]
        v = m["value_cm"]

        # 获取正常上限和下限
        nmax_list = kg.get_objects(s, "has_normal_max")
        nmin_list = kg.get_objects(s, "has_normal_min")

        label = None
        normal_ref = None
        direction = None

        # 上限型：超过normal_max
        if nmax_list:
            nmax = _parse_threshold_value(nmax_list[0])
            if v > nmax:
                # 查找对应的严重程度分级
                for t in kg.get_triples(s, "has_severity_level"):
                    meta = t.get("meta", {})
                    lo = meta.get("min", 0)
                    hi = meta.get("max", 999)
                    if lo <= v < hi:
                        label = t["o"]
                        break
                normal_ref = nmax
                direction = "max"

        # 下限型：低于normal_min
        if nmin_list and label is None:
            nmin = _parse_threshold_value(nmin_list[0])
            if v < nmin:
                # 查找对应的严重程度分级
                for t in kg.get_triples(s, "has_severity_level"):
                    meta = t.get("meta", {})
                    lo = meta.get("min", 0)
                    hi = meta.get("max", 999)
                    if lo <= v < hi:
                        label = t["o"]
                        break
                normal_ref = nmin
                direction = "min"

        # 如果有分级标签，添加到输出
        if label:
            out.append({
                **m,
                "label": label,
                "normal_ref": normal_ref,
                "direction": direction
            })

    return out
