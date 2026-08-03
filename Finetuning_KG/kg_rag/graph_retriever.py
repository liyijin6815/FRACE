"""Retrieve qualitative findings and their graph-linked knowledge."""


NEGATIONS = [
    "未见", "未见明显", "未见明确", "未见确切", "未见异常",
    "无", "正常", "显示清晰", "未显示", "显示欠清", "不宽", "未见扩张",
]


def match_finding_nodes(text, kg):
    """Match Finding aliases while filtering local negation expressions."""
    hits = []
    seen = set()

    for finding_name in kg.entities_by_type("Finding"):
        attr = kg.get_entity(finding_name)
        aliases = sorted(set([finding_name] + attr.get("aliases", [])), key=len, reverse=True)

        for alias in aliases:
            start = 0
            while True:
                index = text.find(alias, start)
                if index == -1:
                    break
                window_start = max(0, index - 10)
                window_end = min(len(text), index + len(alias) + 10)
                surrounding = text[window_start:index] + text[index + len(alias):window_end]
                if not any(negation in surrounding for negation in NEGATIONS):
                    if finding_name not in seen:
                        seen.add(finding_name)
                        hits.append(finding_name)
                    break
                start = index + len(alias)

    return hits


def _interpretation_text(entity_name, kg):
    """Dereference an ``indicates`` node to its clinical text."""
    entity = kg.get_entity(entity_name) or {}
    return entity.get("text") or entity.get("label") or entity_name


def retrieve_finding_knowledge(finding_nodes, kg):
    """Traverse graph relations for interpretation, location, and diagnoses."""
    output = []

    for finding in finding_nodes:
        interpretation_nodes = kg.get_objects(finding, "indicates")
        interpretations = [
            _interpretation_text(entity_name, kg) for entity_name in interpretation_nodes
        ]
        output.append({
            "finding": finding,
            "interpretation": "；".join(text for text in interpretations if text),
            "located_in": kg.get_objects(finding, "located_in"),
            "associated_with": kg.get_objects(finding, "associated_with"),
        })

    return output
