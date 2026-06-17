"""
LLM Evaluation Part 1: Use LLM to extract diagnostic keywords from each output.

Usage:
  DIAG_ROOT=./results/diagnosis LLM_ROOT=./results/llm MODEL_NAME=model \
    python step4-3_evaluate_outputs_with_llm_part1.py

Requires: OPENAI_API_KEY and OPENAI_BASE_URL environment variables.
"""

import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# API configuration - set via environment variables
API_KEY    = os.environ.get('OPENAI_API_KEY', 'your-api-key-here')
BASE_URL   = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1/')
EVAL_MODEL = os.environ.get('EVAL_MODEL', 'gpt-4o')
MAX_WORKERS = 6

# Path configuration
input_root_path  = os.environ.get('DIAG_ROOT', './results/diagnosis')
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
    '你是一名对胎儿影像报告进行"关键词提取"的人工智能助手。'
    '你将会收到一段文本，你的任务是分析这段文本中提到了哪些异常或疾病，并且进行格式化的整理。'
    '你需要遵循以下要求：\n'
    '(1) 只整理异常生理现象和疾病，忽视文本中提及的正常生理现象；'
    '一些较轻微的改变也需要被整理，但要与严重异常有所区分；母体的改变也需要整理。\n'
    '(2) 一项异常指的是"1个部位/器官的1种疾病"，不可随意拆分与合并。\n'
    '(3) 你的输出中应保留分析过程，结尾必须严格按照以下格式：'
    '"综上所述：共发现胎儿严重异常N项：1. XXX，2. XXX，...，N. XXX；'
    '发现胎儿轻微异常及母体相关异常M项：1. YYY，2. YYY，...，M. YYY。"'
)


class FetalMRIEvaluationbyModel():

    def __init__(self, api_key, base_url):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.system_prompt = SYSTEM_PROMPT

    def call_llm(self, model_name, text_content, max_retries=8):
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {'role': 'system', 'content': self.system_prompt},
                        {'role': 'user',   'content': text_content},
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                )
                return completion.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 * attempt)
                else:
                    print(f"Max retries reached. Error: {e}")
                    return None

    def _call_with_null_retry(self, model_name, text_content, null_retries=10):
        for retry in range(null_retries):
            result = self.call_llm(model_name, text_content)
            if result and result.strip():
                return result.strip()
            time.sleep(0.5)
        return None

    def process_single_file(self, input_filename, output_filename, model_name):
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} records")

        results_dict = {}
        lock = threading.Lock()
        success_count = 0
        fail_count = 0
        processed_count = 0

        def _process_one(i, item):
            nonlocal success_count, fail_count, processed_count
            text_content = '请整理以下case：\n"报告文本"：' + item['cleaned_output']
            result = self._call_with_null_retry(model_name, text_content)
            with lock:
                if result:
                    results_dict[i] = {'output': item['cleaned_output'], 'evaluated_result': result}
                    success_count += 1
                else:
                    fail_count += 1
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f'  Processed {processed_count}/{len(data)}...')

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_one, i, item): i for i, item in enumerate(data)}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    with lock:
                        fail_count += 1
                        processed_count += 1

        output_list = [results_dict[i] for i in sorted(results_dict.keys())]
        print(f"  Done: {success_count} success, {fail_count} failed")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)

    def process_label(self, input_filename, output_filename, model_name):
        with open(input_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  Loaded label {len(data)} records")

        results_dict = {}
        lock = threading.Lock()
        success_count = 0
        fail_count = 0
        processed_count = 0

        def _process_one(i, item):
            nonlocal success_count, fail_count, processed_count
            text_content = '请整理以下case：\n"报告文本"：' + item['cleaned_label']
            result = self._call_with_null_retry(model_name, text_content)
            with lock:
                if result:
                    results_dict[i] = {'label': item['cleaned_label'], 'evaluated_result': result}
                    success_count += 1
                else:
                    fail_count += 1
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f'  Processed {processed_count}/{len(data)}...')

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_process_one, i, item): i for i, item in enumerate(data)}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    with lock:
                        fail_count += 1
                        processed_count += 1

        output_list = [results_dict[i] for i in sorted(results_dict.keys())]
        print(f"  Done: {success_count} success, {fail_count} failed")
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    evaluator = FetalMRIEvaluationbyModel(API_KEY, BASE_URL)

    for model_name in file_path_list:
        output_dir = os.path.join(output_root_path, model_name)
        os.makedirs(output_dir, exist_ok=True)

        for dataset_name in dataset_list:
            cleaned_name  = '[Cleaned]_' + dataset_name
            file_full     = os.path.join(input_root_path, model_name, cleaned_name)
            output_full   = os.path.join(output_dir, dataset_name)

            if os.path.exists(output_full):
                print(f'  Already exists, skip: {output_full}')
                continue
            if not os.path.exists(file_full):
                print(f'  Input file missing, skip: {file_full}')
                continue

            print(f'\nProcessing: {model_name} | {cleaned_name}')
            evaluator.process_single_file(file_full, output_full, EVAL_MODEL)

    # Process labels (only need to run once)
    print('\nProcessing labels...')
    label_output_dir = os.path.join(output_root_path, 'label')
    os.makedirs(label_output_dir, exist_ok=True)

    for dataset_name in dataset_list:
        cleaned_name  = '[Cleaned]_' + dataset_name
        file_full     = os.path.join(input_root_path, file_path_list[0], cleaned_name)
        label_full    = os.path.join(label_output_dir, dataset_name)

        if os.path.exists(label_full):
            print(f'  Label already exists, skip: {label_full}')
            continue
        if not os.path.exists(file_full):
            print(f'  Label source missing, skip: {file_full}')
            continue

        evaluator.process_label(file_full, label_full, EVAL_MODEL)

    print('\nAll LLM evaluations (Part 1) completed!')
