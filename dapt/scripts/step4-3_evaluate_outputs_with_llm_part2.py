"""
LLM Evaluation Part 2: Compare extracted keywords to compute TP/FP/FN.

Usage:
  LLM_ROOT=./results/llm MODEL_NAME=model \
    python step4-3_evaluate_outputs_with_llm_part2.py

Requires: OPENAI_API_KEY and OPENAI_BASE_URL environment variables.
"""

import os
import json
import time
import re
import requests

# API configuration - set via environment variables
API_KEY    = os.environ.get('OPENAI_API_KEY', 'your-api-key-here')
BASE_URL   = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1/')
EVAL_MODEL = os.environ.get('EVAL_MODEL', 'gpt-4o')

# Path configuration
input_root_path  = os.environ.get('LLM_ROOT', './results/llm')
output_root_path = os.environ.get('LLM_ROOT', './results/llm')
file_path_list = [os.environ.get('MODEL_NAME', 'model')]

dataset_list = [
    'InDistribution_test.json',
    'OutofDistribution_center1_normal.json',
    'OutofDistribution_center1_patients.json',
    'OutofDistribution_center2_normal.json',
    'OutofDistribution_center2_patients.json',
]

SYSTEM_PROMPT = (
    '你是一名对胎儿影像报告进行"关键词比对"的人工智能助手。'
    '你将会收到一组成对的输入，其中包括Output和Label。'
    '你的任务是分析、比对Output和Label中的各项关键词，'
    '从而生成True Positive (TP)、False Positive (FP)、False Negative (FN) 的混淆矩阵。'
    '关键词的含义相近即可，无需追求文本表述的一致性。'
    '你的输出必须按照以下格式："XXX""XXX"为TP，"XXX""XX"为FP，"XXX""XXX"为FN。'
    '共发现TP=L项，FP=M项，FN=N项。'
)


def preprocess(text):
    text = text.replace('\n', '').replace('\r', '').replace(' ', '').replace('　', '')
    keyword = '综上所述：'
    pos = text.rfind(keyword)
    if pos != -1:
        return text[pos + len(keyword):]
    colon_pos = text.rfind('：')
    if colon_pos != -1:
        return text[colon_pos + 1:]
    return text


class FetalMRIEvaluationbyModel():

    def __init__(self, api_key, base_url):
        self.api_key  = api_key
        self.base_url = base_url
        self.system_prompt = SYSTEM_PROMPT

    def call_llm(self, model_name, text_content, max_retries=8):
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.base_url}chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user",   "content": text_content},
                        ],
                        "temperature": 0.1,
                        "max_tokens": 2048
                    },
                    timeout=120
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                if content:
                    content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL).strip()
                return content
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"  [FAILED] {type(e).__name__}: {e}")
                    return None

    def process_single_file(self, output_data, label_data, output_filename, model_name):
        assert len(output_data) == len(label_data), \
            f'Sample count mismatch: output={len(output_data)}, label={len(label_data)}'

        output_list = []
        success_count = 0
        fail_count = 0

        for idx in range(len(output_data)):
            output_text = preprocess(output_data[idx]['evaluated_result'])
            label_text  = preprocess(label_data[idx]['evaluated_result'])
            text_content = '请比对以下case：\n' + 'Output: "' + output_text + '"\n' \
                         + 'Label: "' + label_text + '"'

            result = self.call_llm(model_name, text_content)
            if result:
                output_list.append({"model_result": result.strip()})
                success_count += 1
            else:
                fail_count += 1

            if (idx + 1) % 10 == 0:
                print(f'  Processed {idx + 1}/{len(output_data)}...')

        print(f"  Done: {success_count} success, {fail_count} failed")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    evaluator = FetalMRIEvaluationbyModel(API_KEY, BASE_URL)

    for model_name in file_path_list:
        output_dir = os.path.join(output_root_path, model_name)
        os.makedirs(output_dir, exist_ok=True)

        for dataset_name in dataset_list:
            output_filename = os.path.join(output_dir, f"[LLM_Eva]_{dataset_name}")
            if os.path.exists(output_filename):
                print(f'  Already exists, skip: {output_filename}')
                continue

            input_filename = os.path.join(input_root_path, model_name, dataset_name)
            local_label    = os.path.join(input_root_path, model_name, 'label', dataset_name)
            label_filename = local_label if os.path.exists(local_label) \
                             else os.path.join(input_root_path, 'label', dataset_name)

            if not os.path.exists(input_filename):
                print(f'  Model input missing, skip: {input_filename}')
                continue
            if not os.path.exists(label_filename):
                print(f'  Label file missing, skip: {label_filename}')
                continue

            with open(input_filename, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            with open(label_filename, 'r', encoding='utf-8') as f:
                label_data = json.load(f)

            print(f'\nProcessing: {model_name} | {dataset_name}')
            evaluator.process_single_file(input_data, label_data, output_filename, EVAL_MODEL)

    print('\nAll LLM evaluations (Part 2) completed!')
