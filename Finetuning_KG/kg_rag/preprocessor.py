"""Normalize report text before KG retrieval."""
import re


def normalize_text(text):
    """Normalize punctuation and measurement units.

    Args:
        text: Raw report text.

    Returns:
        Normalized report text.
    """
    # Convert selected full-width punctuation to ASCII equivalents.
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace('：', ':').replace('，', ',')

    # Normalize unit spelling while preserving Chinese clinical terms.
    text = text.replace('厘米', 'cm').replace('毫米', 'mm')
    text = text.replace('CM', 'cm').replace('MM', 'mm')
    text = text.replace('Cm', 'cm').replace('Mm', 'mm')

    return text


def split_sentences(text):
    """Split report text on Chinese and ASCII sentence boundaries.

    Args:
        text: Normalized report text.

    Returns:
        Non-empty sentence fragments.
    """
    # Chinese punctuation is part of the functional expression.
    sentences = re.split(r'[。；;\n]+', text)
    # Drop empty fragments.
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences
