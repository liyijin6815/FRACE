"""文本预处理模块"""
import re


def normalize_text(text):
    """文本归一化处理

    Args:
        text: 原始文本

    Returns:
        归一化后的文本
    """
    # 全角转半角
    text = text.replace('（', '(').replace('）', ')')
    text = text.replace('：', ':').replace('，', ',')

    # 单位归一化
    text = text.replace('厘米', 'cm').replace('毫米', 'mm')
    text = text.replace('CM', 'cm').replace('MM', 'mm')
    text = text.replace('Cm', 'cm').replace('Mm', 'mm')

    return text


def split_sentences(text):
    """句子切分

    Args:
        text: 文本

    Returns:
        句子列表
    """
    # 按中文句号、分号、换行符切分
    sentences = re.split(r'[。；;\n]+', text)
    # 去除空白句
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences
