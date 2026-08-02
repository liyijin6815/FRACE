"""数值规则化提取模块 - 子句级结构锚定与双侧测量匹配。"""
import re


_CLAUSE_SPLIT_RE = re.compile(r"[，,、。；;\n]+")
_SIDE_RE = re.compile(r"左侧|右侧|双侧|左|右")
_VALUE_RE = re.compile(
    r"(?:宽径约|宽约|深约|长约|约|测值|最宽处约|最宽约为|最宽约|最宽处|"
    r"最宽|最大约|最大横径约|最大横径|宽径|宽|深|长|径)"
    r"\s*(\d+\.?\d*)\s*(cm|mm)",
    re.IGNORECASE,
)
_CONTINUATION_RE = re.compile(
    r"^\s*(?:横轴位|轴位|冠状位|矢状位)?\s*(左侧|右侧|双侧|左|右)"
    r"[^，,、。；;\n]{0,18}?(\d+\.?\d*)\s*(cm|mm)",
    re.IGNORECASE,
)


def to_cm(value, unit):
    """将 mm 统一换算为 cm。"""
    return round(value / 10.0, 4) if unit.lower() == "mm" else round(value, 4)


def _normalize_side(side):
    if side.startswith("左"):
        return "左侧"
    if side.startswith("右"):
        return "右侧"
    if side.startswith("双"):
        return "双侧"
    return ""


def _nearest_side(text, value_start):
    """取当前数值之前、当前局部片段内最近出现的侧别。"""
    local_text = text[:value_start]
    matches = list(_SIDE_RE.finditer(local_text))
    return _normalize_side(matches[-1].group()) if matches else ""


def _extract_anchored_clause(clause, aliases):
    """从含结构别名的单个子句中提取该结构的测量值。"""
    alias_matches = []
    for alias in aliases:
        start = clause.find(alias)
        if start >= 0:
            alias_matches.append((start, -len(alias), alias))
    if not alias_matches:
        return []

    alias_start, _, alias = min(alias_matches)
    search_start = alias_start + len(alias)
    after_alias = clause[search_start:]
    match = _VALUE_RE.search(after_alias)
    matches = [match] if match else []

    results = []
    for match in matches:
        value_start = search_start + match.start()
        side = _nearest_side(clause, value_start)
        results.append((side, float(match.group(1)), match.group(2).lower()))
    return results


def _extract_continuation_clause(clause):
    """提取“左侧宽约.../右侧约...”这类紧邻的双侧续写子句。"""
    match = _CONTINUATION_RE.search(clause)
    if not match:
        return []
    return [(_normalize_side(match.group(1)), float(match.group(2)), match.group(3).lower())]


def _all_measurement_aliases(kg):
    aliases = set()
    for name in kg.entities_by_type("Measurement"):
        attr = kg.get_entity(name)
        aliases.add(name)
        aliases.update(attr.get("aliases", []))
    return aliases


def extract_measurements(text, kg):
    """按局部子句抽取结构、侧别和测量值，避免跨结构串值。"""
    clauses = [part.strip() for part in _CLAUSE_SPLIT_RE.split(text) if part.strip()]
    all_aliases = _all_measurement_aliases(kg)
    results = []
    seen = set()

    for structure in kg.entities_by_type("Measurement"):
        attr = kg.get_entity(structure)
        aliases = sorted(set([structure] + attr.get("aliases", [])), key=len, reverse=True)

        for index, clause in enumerate(clauses):
            values = _extract_anchored_clause(clause, aliases)
            if not values:
                continue

            candidates = [(clause, values)]

            if index + 1 < len(clauses):
                next_clause = clauses[index + 1]
                contains_other_structure = any(
                    alias in next_clause for alias in all_aliases if alias not in aliases
                )
                if not contains_other_structure:
                    continuation = _extract_continuation_clause(next_clause)
                    if continuation:
                        candidates.append((next_clause, continuation))

            for raw_clause, clause_values in candidates:
                for side, value, unit in clause_values:
                    value_cm = to_cm(value, unit)
                    key = (structure, side, value_cm)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append({
                        "structure": structure,
                        "side": side,
                        "value_cm": value_cm,
                        "raw_sentence": raw_clause,
                    })

    return results


def _parse_threshold_value(threshold_str):
    match = re.search(r"(\d+\.?\d*)", threshold_str)
    if match:
        return float(match.group(1))
    return float(threshold_str)


def judge_thresholds(measurements, kg):
    """根据图谱阈值判断测量值的异常方向与严重程度。"""
    output = []

    for measurement in measurements:
        structure = measurement["structure"]
        value_cm = measurement["value_cm"]
        normal_max_values = kg.get_objects(structure, "has_normal_max")
        normal_min_values = kg.get_objects(structure, "has_normal_min")
        label = None
        normal_ref = None
        direction = None

        if normal_max_values:
            normal_max = _parse_threshold_value(normal_max_values[0])
            if value_cm > normal_max:
                for triple in kg.get_triples(structure, "has_severity_level"):
                    meta = triple.get("meta", {})
                    if meta.get("min", 0) <= value_cm < meta.get("max", 999):
                        label = triple["o"]
                        break
                normal_ref = normal_max
                direction = "max"

        if normal_min_values and label is None:
            normal_min = _parse_threshold_value(normal_min_values[0])
            if value_cm < normal_min:
                for triple in kg.get_triples(structure, "has_severity_level"):
                    meta = triple.get("meta", {})
                    if meta.get("min", 0) <= value_cm < meta.get("max", 999):
                        label = triple["o"]
                        break
                normal_ref = normal_min
                direction = "min"

        if label:
            output.append({
                **measurement,
                "label": label,
                "normal_ref": normal_ref,
                "direction": direction,
            })

    return output
