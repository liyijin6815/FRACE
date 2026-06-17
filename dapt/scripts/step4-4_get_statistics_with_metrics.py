"""
Statistics aggregation: combine NLP metric scores + LLM evaluation scores,
output comparison tables.

Usage:
  SCORES_ROOT=./results/scores LLM_ROOT=./results/llm METRICS_ROOT=./results/metrics \
    MODEL_NAME=model python step4-4_get_statistics_with_metrics.py
"""

import os
import re
import json
import numpy as np
import pandas as pd

scores_root_path    = os.environ.get('SCORES_ROOT', './results/scores')
LLM_root_path       = os.environ.get('LLM_ROOT', './results/llm')
output_combined_dir = os.environ.get('METRICS_ROOT', './results/metrics')
os.makedirs(output_combined_dir, exist_ok=True)

_model_name    = os.environ.get('MODEL_NAME', 'model')
file_path_list = [_model_name]
method_names   = [_model_name]

base_dataset_list = [
    'InDistribution_test.json',
    'OutofDistribution_center1_normal.json',
    'OutofDistribution_center1_patients.json',
    'OutofDistribution_center2_normal.json',
    'OutofDistribution_center2_patients.json',
]

center_groups = {
    'OutofDistribution_center1_all': [
        'OutofDistribution_center1_normal.json',
        'OutofDistribution_center1_patients.json',
    ],
    'OutofDistribution_center2_all': [
        'OutofDistribution_center2_normal.json',
        'OutofDistribution_center2_patients.json',
    ],
}

assert len(file_path_list) == len(method_names)

NLP_METRIC_NAMES = ['rouge_f1', 'rouge_p', 'rouge_r', 'bert_f1', 'bert_p', 'bert_r', 'sentence_sim']
LLM_METRIC_NAMES = ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']


def parse_tp_fp_fn(text):
    match = re.search(r'TP=(\d+)项[，,].*?FP=(\d+)项[，,].*?FN=(\d+)项', text)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    tp_m = re.search(r'TP[=＝](\d+)', text)
    fp_m = re.search(r'FP[=＝](\d+)', text)
    fn_m = re.search(r'FN[=＝](\d+)', text)
    if tp_m and fp_m and fn_m:
        return int(tp_m.group(1)), int(fp_m.group(1)), int(fn_m.group(1))
    return 0, 0, 0


def compute_prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def load_nlp_stats():
    mean_dict = {}
    raw_dict  = {}

    for base_name in base_dataset_list:
        rows = []
        raw_dict[base_name] = {}

        for method_idx, method_name in enumerate(method_names):
            model_name = file_path_list[method_idx]
            csv_name   = f'[Scores]_{base_name.replace(".json", "")}.csv'
            full_path  = os.path.join(scores_root_path, model_name, csv_name)

            if not os.path.exists(full_path):
                print(f'  Warning: file not found, skip: {full_path}')
                rows.append([float('nan')] * len(NLP_METRIC_NAMES))
                raw_dict[base_name][method_name] = pd.DataFrame(columns=NLP_METRIC_NAMES)
            else:
                df  = pd.read_csv(full_path)
                raw_dict[base_name][method_name] = df[NLP_METRIC_NAMES]
                avg = list(np.mean(df[NLP_METRIC_NAMES].values, axis=0))
                rows.append(avg)
                print(f'  {method_name} | {base_name}: '
                      f'ROUGE-L={avg[0]:.4f}, BERT-F1={avg[3]:.4f}, SentSim={avg[6]:.4f}')

        mean_dict[base_name] = pd.DataFrame(rows, columns=NLP_METRIC_NAMES, index=method_names)

    return mean_dict, raw_dict


def load_llm_stats():
    any_found = any(
        os.path.exists(os.path.join(LLM_root_path, m, f'[LLM_Eva]_{d.replace(".json", "")}.json'))
        for d in base_dataset_list for m in file_path_list
    )
    if not any_found:
        print('  No LLM evaluation files found, skip Part B.')
        return None, None

    summary_dict = {}
    raw_counts   = {}

    for base_name in base_dataset_list:
        rows = []
        raw_counts[base_name] = {}

        for method_idx, method_name in enumerate(method_names):
            model_name = file_path_list[method_idx]
            json_path  = os.path.join(LLM_root_path, model_name,
                                      f'[LLM_Eva]_{base_name.replace(".json", "")}.json')

            if not os.path.exists(json_path):
                print(f'  Warning: file not found, skip: {json_path}')
                rows.append([0, 0, 0, 0.0, 0.0, 0.0])
                raw_counts[base_name][method_name] = (0, 0, 0)
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            total_tp = total_fp = total_fn = 0
            parse_fail = 0
            for item in data:
                tp, fp, fn = parse_tp_fp_fn(item.get('model_result', ''))
                if tp == 0 and fp == 0 and fn == 0 and item.get('model_result', '').strip():
                    parse_fail += 1
                total_tp += tp; total_fp += fp; total_fn += fn

            if parse_fail > 0:
                print(f'  Warning: {method_name} | {base_name}: {parse_fail}/{len(data)} parse failures')

            precision, recall, f1 = compute_prf(total_tp, total_fp, total_fn)
            rows.append([total_tp, total_fp, total_fn, precision, recall, f1])
            raw_counts[base_name][method_name] = (total_tp, total_fp, total_fn)
            print(f'  {method_name} | {base_name}: '
                  f'TP={total_tp}, FP={total_fp}, FN={total_fn}  '
                  f'P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}')

        summary_dict[base_name] = pd.DataFrame(rows, columns=LLM_METRIC_NAMES, index=method_names)

    return summary_dict, raw_counts


def build_center_combined(nlp_raw, llm_raw_counts):
    result = {}
    for center_label, sub_datasets in center_groups.items():
        print(f'\n  Combining {center_label}: {sub_datasets}')

        nlp_rows = []
        for method_name in method_names:
            frames = [nlp_raw[ds][method_name] for ds in sub_datasets
                      if ds in nlp_raw and method_name in nlp_raw[ds]
                      and not nlp_raw[ds][method_name].empty]
            avg = list(np.nanmean(pd.concat(frames, ignore_index=True).values, axis=0)) \
                  if frames else [float('nan')] * len(NLP_METRIC_NAMES)
            nlp_rows.append(avg)
        df_nlp = pd.DataFrame(nlp_rows, columns=NLP_METRIC_NAMES, index=method_names)

        df_llm = None
        if llm_raw_counts is not None:
            llm_rows = []
            for method_name in method_names:
                sum_tp = sum_fp = sum_fn = 0
                for ds in sub_datasets:
                    if ds in llm_raw_counts and method_name in llm_raw_counts[ds]:
                        tp, fp, fn = llm_raw_counts[ds][method_name]
                        sum_tp += tp; sum_fp += fp; sum_fn += fn
                precision, recall, f1 = compute_prf(sum_tp, sum_fp, sum_fn)
                llm_rows.append([sum_tp, sum_fp, sum_fn, precision, recall, f1])
            df_llm = pd.DataFrame(llm_rows, columns=LLM_METRIC_NAMES, index=method_names)

        result[center_label] = pd.concat([df_nlp, df_llm], axis=1) if df_llm is not None else df_nlp

    return result


def save_combined_csv(nlp_mean, llm_summary):
    for base_name in base_dataset_list:
        df_nlp = nlp_mean[base_name]
        df_combined = pd.concat([df_nlp, llm_summary[base_name]], axis=1) \
                      if llm_summary and base_name in llm_summary else df_nlp
        out_csv = os.path.join(output_combined_dir,
                               f'[Combined]_{base_name.replace(".json", "")}.csv')
        df_combined.to_csv(out_csv, encoding='utf-8-sig')
        print(f'  Saved: {out_csv}')


def save_center_csv(center_dict):
    for center_label, df in center_dict.items():
        out_csv = os.path.join(output_combined_dir, f'[Combined]_{center_label}.csv')
        df.to_csv(out_csv, encoding='utf-8-sig')
        print(f'  Saved: {out_csv}')


if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('Statistics Aggregation')
    print('=' * 60)

    print('\n--- Part A: NLP Metrics ---')
    nlp_mean, nlp_raw = load_nlp_stats()

    print('\n--- Part B: LLM Metrics ---')
    llm_summary, llm_raw_counts = load_llm_stats()

    print('\n--- Saving per-dataset Combined CSV ---')
    save_combined_csv(nlp_mean, llm_summary)

    print('\n--- Per-center Aggregation ---')
    center_dict = build_center_combined(nlp_raw, llm_raw_counts)
    save_center_csv(center_dict)

    print('\nAll statistics completed!')
