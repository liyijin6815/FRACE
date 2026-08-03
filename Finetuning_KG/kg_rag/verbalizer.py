"""Verbalize structured KG retrieval results."""


def verbalize_numeric(judgement, kg):
    """Render one numeric judgement with its KG template.

    Args:
        judgement: One result from ``judge_thresholds``.
        kg: Loaded ``ClinicalKG`` instance.

    Returns:
        Clinical reference sentence.
    """
    attr = kg.get_entity(judgement["structure"])
    template = attr.get("template", "")

    if not template:
        return ""

    # Populate the placeholders shared by the KG templates.
    params = {
        "side": judgement.get("side", ""),
        "name": judgement["structure"],
        "value": round(judgement["value_cm"], 2),
        "label": judgement["label"]
    }

    # Add the applicable normal reference bound.
    if judgement.get("direction") == "max":
        params["normal_max"] = judgement.get("normal_ref")
    elif judgement.get("direction") == "min":
        params["normal_min"] = judgement.get("normal_ref")

    try:
        return template.format(**params)
    except KeyError:
        # Skip malformed templates without breaking dataset construction.
        return ""


def verbalize_finding(item):
    """Render one qualitative finding and its differential diagnoses.

    Args:
        item: One result from ``retrieve_finding_knowledge``.

    Returns:
        Clinical reference sentence.
    """
    s = item["interpretation"]

    # The Chinese connectors are part of the model-facing clinical text.
    if item["associated_with"]:
        s += "，需与" + "、".join(item["associated_with"]) + "鉴别"

    return s
