"""
Post-prediction text extraction and cleaning.
Extracts model outputs from generated_predictions.jsonl, removes think tags,
and applies rule-based text cleaning.

Usage:
  PRED_DIR=./predictions/v1/sft_only DIAG_DIR=./results/model python step4-1_text_extract_and_clean.py
"""

import re
import json
import os
from pathlib import Path

REMOVE_THINK_TAG = True

DATASET_NAMES = [
    "InDistribution_test",
    "OutofDistribution_center1_normal",
    "OutofDistribution_center1_patients",
    "OutofDistribution_center2_normal",
    "OutofDistribution_center2_patients"
]

BASE_INPUT_DIR  = os.environ.get('PRED_DIR', './predictions')
BASE_OUTPUT_DIR = os.environ.get('DIAG_DIR', './results/model')


def step_a_extract(input_path):
    output_list = []

    with open(input_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    output_list.append({
                        'output': item.get('predict', item.get('model_output', '')),
                        'label':  item.get('label', item.get('reference_output', '')),
                    })
                return output_list
        except json.JSONDecodeError:
            f.seek(0)

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                output_list.append({
                    'output': item.get('predict', item.get('model_output', '')),
                    'label':  item.get('label', item.get('reference_output', '')),
                })
            except json.JSONDecodeError:
                pass

    return output_list


def strip_think_tag(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _remove_between_tags(text, start_tag, end_tag):
    pattern = re.escape(start_tag) + r'.*?' + re.escape(end_tag)
    return re.sub(pattern, '', text, flags=re.DOTALL)


def rule_based_data_clean(text):
    if "【最终诊断意见】" in text:
        text = text.split("【最终诊断意见】")[-1]

    text = re.sub(r'[•\s]*\(?\d+\)?[.、）)]', '', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r' *([，。；：！？、…—～·《》「」【】〔〕]) *', r'\1', text)
    punctuation_map = {',': '，', '.': '。', ';': '；', ':': '：', '(': '（', ')': '）'}
    for eng, chn in punctuation_map.items():
        text = text.replace(eng, chn)
    text = re.sub(r'["\'""]', '', text)
    text = text.replace(' ', '')

    term_map = {
        "未见明确异常":    "未见异常",
        "结合产前诊断咨询": "结合产前诊断",
        "子宫体":         "子宫",
        "（末次月经龄）":   "",
    }
    for term, replacement in term_map.items():
        text = text.replace(term, replacement)

    boilerplate_tags = {"针对": "扫描："}
    for start_tag, end_tag in boilerplate_tags.items():
        text = _remove_between_tags(text, start_tag, end_tag)

    return text.strip()


def step_b_clean(raw_list, remove_think=True):
    cleaned_list = []
    for idx, item in enumerate(raw_list):
        process_output = item['output']
        if remove_think:
            process_output = strip_think_tag(process_output)

        cleaned_list.append({
            'patient_id':       idx,
            'original_output':  item['output'],
            'original_label':   item['label'],
            'cleaned_output':   rule_based_data_clean(process_output),
            'cleaned_label':    rule_based_data_clean(item['label']),
        })
    return cleaned_list


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    success_count = 0

    for dataset_name in DATASET_NAMES:
        print(f"========== Processing: {dataset_name} ==========")

        input_jsonl       = os.path.join(BASE_INPUT_DIR, dataset_name, 'generated_predictions.jsonl')
        raw_output_json   = os.path.join(BASE_OUTPUT_DIR, f"[Raw]_{dataset_name}.json")
        clean_output_json = os.path.join(BASE_OUTPUT_DIR, f"[Cleaned]_{dataset_name}.json")

        if not os.path.exists(input_jsonl):
            print(f'  Input file not found, skip: {input_jsonl}\n')
            continue

        raw_list = step_a_extract(input_jsonl)
        save_json(raw_list, raw_output_json)
        print(f"  Extracted {len(raw_list)} records")

        cleaned_list = step_b_clean(raw_list, remove_think=REMOVE_THINK_TAG)
        save_json(cleaned_list, clean_output_json)
        print(f"  Cleaning done\n")
        success_count += 1

    print(f"Batch processing complete: {success_count}/{len(DATASET_NAMES)} datasets.")


if __name__ == '__main__':
    main()
