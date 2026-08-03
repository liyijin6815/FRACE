"""Build a clinical knowledge reference from numeric and graph retrieval."""
from .preprocessor import normalize_text
from .numeric_extractor import extract_measurements, judge_thresholds
from .graph_retriever import match_finding_nodes, retrieve_finding_knowledge
from .verbalizer import verbalize_numeric, verbalize_finding


def build_kg_context(report_text, kg):
    """Build the KG context for one imaging-findings report.

    Args:
        report_text: Imaging-findings text.
        kg: Loaded ``ClinicalKG`` instance.

    Returns:
        A formatted clinical reference, or an empty string when nothing matches.
    """
    # Normalize punctuation and units before both retrieval paths.
    text = normalize_text(report_text)

    lines = []

    # Path A: clause-level numeric extraction and threshold grading.
    measures = extract_measurements(text, kg)
    judgements = judge_thresholds(measures, kg)
    for j in judgements:
        line = verbalize_numeric(j, kg)
        if line:
            lines.append(line)

    # Path B: alias matching and typed graph traversal.
    finding_nodes = match_finding_nodes(text, kg)
    finding_knowledge = retrieve_finding_knowledge(finding_nodes, kg)
    for item in finding_knowledge:
        line = verbalize_finding(item)
        if line:
            lines.append(line)

    # Deduplicate while preserving retrieval order.
    seen = set()
    dedup = []
    for ln in lines:
        if ln and ln not in seen:
            seen.add(ln)
            dedup.append(ln)

    # Do not append an empty reference block.
    if not dedup:
        return ""

    # This marker is part of the model input format.
    return "【临床知识参考】\n" + "\n".join("- " + x for x in dedup)
