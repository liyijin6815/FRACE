"""Extract LLaMA-Factory predictions and apply rule-based text cleaning."""

import re
import json
import os
from pathlib import Path

# Set REMOVE_THINK_TAG=true for model outputs that contain reasoning tags.
REMOVE_THINK_TAG = os.environ.get('REMOVE_THINK_TAG', 'false').lower() == 'true'

# Configure portable input and output roots through environment variables.
BASE_INPUT_DIR = os.environ.get('PREDICTIONS_ROOT', './predictions')
BASE_OUTPUT_DIR = os.environ.get('CLEANED_ROOT', './test_results_diagnosis/model')

# Dataset identifiers shared by all evaluation stages.
DATASETS = [
    'InDistribution_test',
    'OutofDistribution_center1_normal',
    'OutofDistribution_center1_patients',
    'OutofDistribution_center2_normal',
    'OutofDistribution_center2_patients'
]

# Core transformations
def step_a_extract(input_path):
    """Read JSONL records and retain the raw ``predict`` and ``label`` fields."""
    output_list = []
    skipped = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                output_list.append({
                    'output': item.get('predict', ''),
                    'label':  item.get('label', ''),
                })
            except json.JSONDecodeError as e:
                print(f'    [Step A] Invalid JSON on line {line_num}; skipped: {e}')
                skipped += 1

    print(f'    [Step A] Extracted {len(output_list)} records; skipped {skipped}')
    return output_list


def strip_think_tag(text):
    """Remove a complete multiline ``<think>...</think>`` block."""
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return result.strip()


def _remove_between_tags(text, start_tag, end_tag):
    """Remove a multiline block including its start and end markers."""
    pattern = re.escape(start_tag) + r'.*?' + re.escape(end_tag)
    return re.sub(pattern, '', text, flags=re.DOTALL)


def rule_based_data_clean(text):
    """Apply the report-normalization rules used in evaluation."""
    # Remove numbered-list markers.
    text = re.sub(r'[•\s]*\(?\d+\)?[.、）)]', '', text)

    # Normalize spacing and punctuation.
    text = re.sub(r' +', ' ', text)
    text = re.sub(r' *([，。；：！？、…—～·《》「」【】〔〕]) *', r'\1', text)
    
    punctuation_map = {',': '，', '.': '。', ';': '；', ':': '：', '(': '（', ')': '）'}
    for eng, chn in punctuation_map.items():
        text = text.replace(eng, chn)
        
    text = re.sub(r'["\'“”]', '', text) 
    text = text.replace(' ', '')
    
    # Normalize frequent clinical expressions.
    term_map = {
        "未见明确异常":   "未见异常",
        "结合产前诊断咨询": "结合产前诊断",
        "子宫体":        "子宫",
        "（末次月经龄）":  "",
    }
    for term, replacement in term_map.items():
        text = text.replace(term, replacement)

    # Remove non-clinical formatting boilerplate.
    boilerplate_tags = {"针对": "扫描："}
    for start_tag, end_tag in boilerplate_tags.items():
        text = _remove_between_tags(text, start_tag, end_tag)

    return text.strip()


def step_b_clean(raw_list, remove_think=True):
    """Clean predictions and labels while retaining the original text."""
    cleaned_list = []
    for idx, item in enumerate(raw_list):
        original_output = item['output']
        original_label = item['label']

        process_output = original_output
        if remove_think:
            process_output = strip_think_tag(process_output)
    
        cleaned_output = rule_based_data_clean(process_output)
        cleaned_label = rule_based_data_clean(original_label)

        cleaned_list.append({
            'patient_id': idx,
            'original_output': original_output,
            'original_label': original_label,
            'cleaned_output': cleaned_output,
            'cleaned_label': cleaned_label
        })

    print(f'    [Step B] Cleaned {len(cleaned_list)} records (remove_think={remove_think})')
    return cleaned_list


# File utilities
def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'    Saved -> {path}')


def preview_data(raw_data, cleaned_data, n=1):
    """Print a compact preview during batch processing."""
    print(f'\n    --- Preview (first {n} records) ---')
    for i in range(min(n, len(raw_data))):
        print(f'    [Raw Output]     {raw_data[i]["output"][:80]}...')
        print(f'    [Cleaned Output] {cleaned_data[i]["cleaned_output"][:80]}...')
    print('    ---------------------------\n')


# Main batch pipeline
def main():
    print("Starting prediction post-processing...\n")
    
    # Create the shared output root once.
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    for dataset in DATASETS:
        print(f"========== 正在处理数据集: {dataset} ==========")
        
        # Build paths for one dataset.
        input_jsonl = os.path.join(BASE_INPUT_DIR, dataset, 'generated_predictions.jsonl')
        raw_output_json = os.path.join(BASE_OUTPUT_DIR, f'[Raw]_{dataset}.json')
        cleaned_output_json = os.path.join(BASE_OUTPUT_DIR, f'[Cleaned]_{dataset}.json')

        # Missing predictions are skipped in multi-dataset runs.
        if not os.path.exists(input_jsonl):
            print(f'    ⚠️ 警告: 输入文件不存在, 跳过: {input_jsonl}\n')
            continue

        # Step A: extract JSONL predictions.
        print('    [Step A] 正在提取 predict / label ...')
        raw_list = step_a_extract(input_jsonl)
        save_json(raw_list, raw_output_json)

        # Step B: normalize predictions and labels.
        print('    [Step B] 正在执行文本清洗 ...')
        cleaned_list = step_b_clean(raw_list, remove_think=REMOVE_THINK_TAG)
        save_json(cleaned_list, cleaned_output_json)

        # Print one short preview without flooding the console.
        preview_data(raw_list, cleaned_list, n=1)

    print('🎉 所有数据集批量处理完毕！')


if __name__ == '__main__':
    main()
