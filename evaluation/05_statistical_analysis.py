"""Aggregate NLP and LLM-based metrics with confidence intervals.

The script reports one-sided paired tests for the directional hypothesis that
FetalScribe outperforms each comparison method. NLP metrics use a Wilcoxon
signed-rank test; clinical precision, recall, and F1 use paired permutations.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# Shared configuration
scores_root_path = os.environ.get('SCORES_ROOT', './test_results_scores')
LLM_root_path = os.environ.get('LLM_ROOT', './test_results_LLM')
output_combined_dir = os.environ.get('METRICS_ROOT', './test_results_metrics')
os.makedirs(output_combined_dir, exist_ok=True)

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
PERMUTATION_N = 10000

file_path_list = [
    'finetuning/deepseek-r1-qwen3-8b',
    'finetuning_sr/deepseek-r1-qwen3-8b',
    'finetuning_kg_v3/deepseek-r1-qwen3-8b',
    'finetuning_DAPT/deepseek-r1-qwen3-8b',
    'finetuning_ICL/deepseek-r1-qwen3-8b',
    'finetuning_combined_ICL_ICL/deepseek-r1-qwen3-8b',
    'FetalScribe/deepseek-r1-qwen3-8b',

    'with_ICL/closed_source/gpt-5.1',
    'with_ICL/closed_source/gpt-5.2',
    'with_ICL/closed_source/gpt-5.4',
    'with_ICL/closed_source/gemini-3.1-flash-lite-preview',
    'with_ICL/closed_source/gemini-3.1-pro-preview',
    'with_ICL/closed_source/claude-sonnet-4-6',
    'with_ICL/closed_source/claude-opus-4-6',
    'with_ICL/closed_source/claude-opus-4-7',
    'with_ICL/closed_source/grok-4.1',
    'with_ICL/closed_source/grok-4.2-nothinking',
    'with_ICL/closed_source/doubao-seed-2-0-lite-260215',
    'with_ICL/closed_source/qwen3-max',
    'with_ICL/open_source/qwen3-235b-a22b',
    'with_ICL/open_source/DeepSeek-V3.2',
    'with_ICL/open_source/kimi-k2.5',
    'with_ICL/open_source/glm-5',
    'with_ICL/AntAngelMed',
    'with_ICL/HuatuoGPT-Vision-34B',
    'with_ICL/OpenBioLLM-70B-4bit',

    'without_ICL/closed_source/gpt-5.1',
    'without_ICL/closed_source/gpt-5.2',
    'without_ICL/closed_source/gpt-5.4',
    'without_ICL/closed_source/gemini-3.1-flash-lite-preview',
    'without_ICL/closed_source/gemini-3.1-pro-preview',
    'without_ICL/closed_source/claude-sonnet-4-6',
    'without_ICL/closed_source/claude-opus-4-6',
    'without_ICL/closed_source/claude-opus-4-7',
    'without_ICL/closed_source/grok-4.1',
    'without_ICL/closed_source/grok-4.2-nothinking',
    'without_ICL/closed_source/doubao-seed-2-0-lite-260215',
    'without_ICL/closed_source/qwen3-max',
    'without_ICL/open_source/qwen3-235b-a22b',
    'without_ICL/open_source/DeepSeek-V3.2',
    'without_ICL/open_source/kimi-k2.5',
    'without_ICL/open_source/glm-5',
    'without_ICL/AntAngelMed',
    'without_ICL/HuatuoGPT-Vision-34B',
    'without_ICL/OpenBioLLM-70B-4bit',
]

method_names = [
    'fine-tuned',
    'fine-tuned-sr',
    'fine-tuned-kg_v3',
    'fine-tuned_DAPT',
    'fine-tuned_ICL',
    'fine-tuned_combined_ICL',
    'FetalScribe',

    'with_ICL_gpt-5.1',
    'with_ICL_gpt-5.2',
    'with_ICL_gpt-5.4',
    'with_ICL_gemini-3.1-flash-lite-preview',
    'with_ICL_gemini-3.1-pro-preview',
    'with_ICL_claude-sonnet-4-6',
    'with_ICL_claude-opus-4-6',
    'with_ICL_claude-opus-4-7',
    'with_ICL_grok-4.1',
    'with_ICL_grok-4.2-nothinking',
    'with_ICL_doubao-seed-2-0-lite-260215',
    'with_ICL_qwen3-max',
    'with_ICL_qwen3-235b-a22b',
    'with_ICL_DeepSeek-V3.2',
    'with_ICL_kimi-k2.5',
    'with_ICL_glm-5',
    'with_ICL_AntAngelMed',
    'with_ICL_HuatuoGPT-Vision-34B',
    'with_ICL_OpenBioLLM-70B-4bit',

    'without_ICL_gpt-5.1',
    'without_ICL_gpt-5.2',
    'without_ICL_gpt-5.4',
    'without_ICL_gemini-3.1-flash-lite-preview',
    'without_ICL_gemini-3.1-pro-preview',
    'without_ICL_claude-sonnet-4-6',
    'without_ICL_claude-opus-4-6',
    'without_ICL_claude-opus-4-7',
    'without_ICL_grok-4.1',
    'without_ICL_grok-4.2-nothinking',
    'without_ICL_doubao-seed-2-0-lite-260215',
    'without_ICL_qwen3-max',
    'without_ICL_qwen3-235b-a22b',
    'without_ICL_DeepSeek-V3.2',
    'without_ICL_kimi-k2.5',
    'without_ICL_glm-5',
    'without_ICL_AntAngelMed',
    'without_ICL_HuatuoGPT-Vision-34B',
    'without_ICL_OpenBioLLM-70B-4bit',
]

base_dataset_list = [
    'InDistribution_test.json',
    'OutofDistribution_center1_normal.json',
    'OutofDistribution_center1_patients.json',
    'OutofDistribution_center2_normal.json',
    'OutofDistribution_center2_patients.json'
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

FETALSCRIBE_NAME = 'FetalScribe'

assert len(file_path_list) == len(method_names), \
    'file_path_list 和 method_names 长度必须一致！'


# Metric and formatting utilities
NLP_METRIC_NAMES = ['rouge_p', 'rouge_r', 'rouge_f1',
                    'bert_p', 'bert_r', 'bert_f1', 'sentence_sim']
LLM_METRIC_NAMES = ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
LLM_CI_METRICS = ['Precision', 'Recall', 'F1']

# Metrics included in paired significance tests.
SIG_NLP_METRICS = ['rouge_p', 'rouge_r', 'rouge_f1',
                   'bert_p', 'bert_r', 'bert_f1', 'sentence_sim']
SIG_LLM_METRICS = ['Precision', 'Recall', 'F1']
ALL_SIG_METRICS = SIG_NLP_METRICS + SIG_LLM_METRICS

# Ordered p-value and significance-marker columns.
SIG_P_COLS = [f'{m}_p_vs_FetalScribe' for m in ALL_SIG_METRICS]
SIG_STAR_COLS = [f'{m}_sig' for m in ALL_SIG_METRICS]


def dataset_to_csv_name(name):
    return name.replace('.json', '')


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
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def fmt_ci(lo, hi):
    if np.isnan(lo) or np.isnan(hi):
        return '[nan, nan]'
    return f'[{lo:.3f}, {hi:.3f}]'


def p_to_stars(p):
    """Convert a p value to a publication-table significance marker."""
    if np.isnan(p):
        return '-'
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def bootstrap_nlp_ci(values, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float('nan'), float('nan'), float('nan')
    mean = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    boot = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return mean, lo, hi


def bootstrap_prf_ci(per_sample_tps, per_sample_fps, per_sample_fns,
                     n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    arr = np.array(list(zip(per_sample_tps, per_sample_fps, per_sample_fns)), dtype=float)
    if len(arr) == 0:
        nan3 = (float('nan'), float('nan'), float('nan'))
        return nan3, nan3, nan3

    total_tp = arr[:, 0].sum()
    total_fp = arr[:, 1].sum()
    total_fn = arr[:, 2].sum()
    p_mean, r_mean, f1_mean = compute_prf(total_tp, total_fp, total_fn)

    rng = np.random.default_rng(seed)
    p_boot, r_boot, f1_boot = [], [], []
    for _ in range(n):
        idx = rng.integers(0, len(arr), size=len(arr))
        s = arr[idx].sum(axis=0)
        p, r, f = compute_prf(s[0], s[1], s[2])
        p_boot.append(p)
        r_boot.append(r)
        f1_boot.append(f)

    def ci(mean, boot):
        return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    return ci(p_mean, p_boot), ci(r_mean, r_boot), ci(f1_mean, f1_boot)


def paired_wilcoxon_test_onesided(arr_a, arr_b):
    """Run a one-sided paired Wilcoxon test: arr_a > arr_b."""
    arr_a = np.array(arr_a, dtype=float)
    arr_b = np.array(arr_b, dtype=float)
    valid = ~(np.isnan(arr_a) | np.isnan(arr_b))
    arr_a = arr_a[valid]
    arr_b = arr_b[valid]

    if len(arr_a) < 5:
        return float('nan')

    diff = arr_a - arr_b
    if np.all(diff == 0):
        return 1.0

    try:
        _, p_val = scipy_stats.wilcoxon(arr_a, arr_b, alternative='greater')
        return p_val
    except Exception:
        return float('nan')


def paired_permutation_test_micro_onesided(counts_a, counts_b, metric='F1',
                                           n_perm=PERMUTATION_N, seed=BOOTSTRAP_SEED):
    """Run a one-sided paired permutation test on a micro-averaged metric."""
    if not counts_a or not counts_b:
        return float('nan')
    if len(counts_a) != len(counts_b):
        return float('nan')

    n_samples = len(counts_a)
    arr_a = np.array(counts_a, dtype=float)
    arr_b = np.array(counts_b, dtype=float)

    metric_idx = {'Precision': 0, 'Recall': 1, 'F1': 2}[metric]

    def get_metric(arr):
        tp, fp, fn = arr[:, 0].sum(), arr[:, 1].sum(), arr[:, 2].sum()
        prf = compute_prf(tp, fp, fn)
        return prf[metric_idx]

    # Observed difference: FetalScribe minus the comparison method.
    obs_diff = get_metric(arr_a) - get_metric(arr_b)

    # A non-positive observed difference cannot support the one-sided claim.
    if obs_diff <= 0:
        return 1.0

    rng = np.random.default_rng(seed)
    count_extreme = 0
    for _ in range(n_perm):
        swap_mask = rng.random(n_samples) < 0.5
        perm_a = np.where(swap_mask[:, None], arr_b, arr_a)
        perm_b = np.where(swap_mask[:, None], arr_a, arr_b)
        perm_diff = get_metric(perm_a) - get_metric(perm_b)
        # One-sided tail probability.
        if perm_diff >= obs_diff:
            count_extreme += 1

    return (count_extreme + 1) / (n_perm + 1)


# NLP metric loading
def load_nlp_stats():
    stats_dict = {}
    raw_dict = {}

    for base_name in base_dataset_list:
        stats = []
        raw_dict[base_name] = {}

        for method_idx, method_name in enumerate(method_names):
            model_name = file_path_list[method_idx]
            dataset_csv = f'[Scores]_{base_name.replace(".json", "")}.csv'
            full_path = os.path.join(scores_root_path, model_name, dataset_csv)

            if not os.path.exists(full_path):
                print(f'  警告: 文件不存在，跳过: {full_path}')
                means = [float('nan')] * len(NLP_METRIC_NAMES)
                cis = [(float('nan'), float('nan'))] * len(NLP_METRIC_NAMES)
                raw_dict[base_name][method_name] = pd.DataFrame(columns=NLP_METRIC_NAMES)
            else:
                df = pd.read_csv(full_path)
                raw_dict[base_name][method_name] = df[NLP_METRIC_NAMES]
                means = []
                cis = []
                for col in NLP_METRIC_NAMES:
                    mean, lo, hi = bootstrap_nlp_ci(df[col].values)
                    means.append(mean)
                    cis.append((lo, hi))
                print(f'  {method_name} | {base_name}: ROUGE-P≈{means[0]:.4f}')

            stats.append((method_name, means, cis))

        stats_dict[base_name] = stats

    return stats_dict, raw_dict


# LLM-based clinical metric loading
def load_llm_stats():
    any_found = any(
        os.path.exists(os.path.join(
            LLM_root_path, m,
            f'[LLM_Eva]_{d.replace(".json", "")}.json'))
        for d in base_dataset_list
        for m in file_path_list
    )
    if not any_found:
        print('  未找到 LLM 评估文件，跳过 Part B。')
        return None, None

    stats_dict = {}
    raw_counts = {}

    for base_name in base_dataset_list:
        stats = []
        raw_counts[base_name] = {}

        for method_idx, method_name in enumerate(method_names):
            model_name = file_path_list[method_idx]
            json_path = os.path.join(
                LLM_root_path, model_name,
                f'[LLM_Eva]_{base_name.replace(".json", "")}.json')

            if not os.path.exists(json_path):
                print(f'  警告: 文件不存在，跳过: {json_path}')
                raw_counts[base_name][method_name] = []
                stats.append((method_name, None, None))
                continue

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            per_sample = []
            parse_fail = 0
            for item in data:
                text = item.get('model_result', '')
                tp, fp, fn = parse_tp_fp_fn(text)
                if tp == 0 and fp == 0 and fn == 0 and text.strip():
                    parse_fail += 1
                per_sample.append((tp, fp, fn))

            if parse_fail > 0:
                print(f'  警告: {method_name} | {base_name}: '
                      f'{parse_fail}/{len(data)} 条解析失败')

            raw_counts[base_name][method_name] = per_sample

            tps = [x[0] for x in per_sample]
            fps = [x[1] for x in per_sample]
            fns = [x[2] for x in per_sample]
            total_tp, total_fp, total_fn = sum(tps), sum(fps), sum(fns)

            (p_mean, p_lo, p_hi), (r_mean, r_lo, r_hi), (f1_mean, f1_lo, f1_hi) = \
                bootstrap_prf_ci(tps, fps, fns)

            llm_vals = {'TP': total_tp, 'FP': total_fp, 'FN': total_fn}
            llm_cis = [(p_mean, p_lo, p_hi), (r_mean, r_lo, r_hi), (f1_mean, f1_lo, f1_hi)]
            stats.append((method_name, llm_vals, llm_cis))

            print(f'  {method_name} | {base_name}: '
                  f'TP={total_tp} FP={total_fp} FN={total_fn}  '
                  f'P={p_mean:.4f} R={r_mean:.4f} F1={f1_mean:.4f}')

        stats_dict[base_name] = stats

    return stats_dict, raw_counts


# Paired significance tests
def compute_significance_columns(nlp_raw_for_dataset, llm_raw_for_dataset):
    """Compute paired one-sided p values against FetalScribe for each method."""
    sig_pvals = {m: {} for m in method_names}

    # Load the FetalScribe reference method.
    fs_nlp = nlp_raw_for_dataset.get(FETALSCRIBE_NAME, pd.DataFrame())
    fs_llm = llm_raw_for_dataset.get(FETALSCRIBE_NAME, []) if llm_raw_for_dataset else []

    # Align FetalScribe NLP values by patient identifier.
    fs_nlp_cols = {}
    for col in SIG_NLP_METRICS:
        if not fs_nlp.empty and col in fs_nlp.columns:
            fs_nlp_cols[col] = fs_nlp[col].values
        else:
            fs_nlp_cols[col] = np.array([])

    for method_name in method_names:
        if method_name == FETALSCRIBE_NAME:
            for metric in ALL_SIG_METRICS:
                sig_pvals[method_name][metric] = float('nan')
            continue

        # NLP metrics: one-sided paired Wilcoxon test.
        m_nlp = nlp_raw_for_dataset.get(method_name, pd.DataFrame())
        for col in SIG_NLP_METRICS:
            fs_arr = fs_nlp_cols[col]
            if not m_nlp.empty and col in m_nlp.columns:
                m_arr = m_nlp[col].values
            else:
                m_arr = np.array([])

            if len(fs_arr) > 0 and len(m_arr) > 0 and len(fs_arr) == len(m_arr):
                sig_pvals[method_name][col] = paired_wilcoxon_test_onesided(fs_arr, m_arr)
            else:
                sig_pvals[method_name][col] = float('nan')

        # Clinical metrics: one-sided paired permutation test.
        m_llm = llm_raw_for_dataset.get(method_name, []) if llm_raw_for_dataset else []
        if fs_llm and m_llm and len(fs_llm) == len(m_llm):
            for metric in SIG_LLM_METRICS:
                p_val = paired_permutation_test_micro_onesided(fs_llm, m_llm, metric=metric)
                sig_pvals[method_name][metric] = p_val
                print(f'    Permutation(1-sided) {metric}: '
                      f'{FETALSCRIBE_NAME} vs {method_name} -> p={p_val:.6f}')
        else:
            for metric in SIG_LLM_METRICS:
                sig_pvals[method_name][metric] = float('nan')

    return sig_pvals


# Two-row mean and confidence-interval table
def build_two_row_df(nlp_stats, llm_stats=None, sig_pvals=None):
    """
    nlp_stats: list of (method_name, means_list, cis_list)
    llm_stats: list of (method_name, llm_vals_or_None, llm_cis_or_None)
    sig_pvals: dict{ method_name: dict{ metric: p_value } }
    """
    # Stable publication-table column order.
    all_cols = [''] + NLP_METRIC_NAMES
    if llm_stats is not None:
        all_cols += LLM_METRIC_NAMES
    all_cols += SIG_P_COLS + SIG_STAR_COLS

    records = []
    for i, (method_name, means, cis) in enumerate(nlp_stats):
        llm_v, llm_c = None, None
        if llm_stats is not None:
            llm_v = llm_stats[i][1]
            llm_c = llm_stats[i][2]

        if cis and not all(np.isnan(lo) for lo, hi in cis):
            lo_list = [lo for lo, hi in cis]
            hi_list = [hi for lo, hi in cis]
        else:
            lo_list = [float('nan')] * len(NLP_METRIC_NAMES)
            hi_list = [float('nan')] * len(NLP_METRIC_NAMES)

        # Point-estimate row.
        mean_row = {'': method_name}
        ci_row = {'': ''}

        for col, mean, lo, hi in zip(NLP_METRIC_NAMES,
                                     means if means else [float('nan')] * len(NLP_METRIC_NAMES),
                                     lo_list, hi_list):
            mean_row[col] = round(mean, 3) if not np.isnan(mean) else ''
            ci_row[col] = fmt_ci(lo, hi)

        if llm_v is not None:
            mean_row['TP'] = llm_v['TP']
            mean_row['FP'] = llm_v['FP']
            mean_row['FN'] = llm_v['FN']
            ci_row['TP'] = ''
            ci_row['FP'] = ''
            ci_row['FN'] = ''
            for metric, (m, lo, hi) in zip(LLM_CI_METRICS, llm_c):
                mean_row[metric] = round(m, 3) if not np.isnan(m) else ''
                ci_row[metric] = fmt_ci(lo, hi)
        elif llm_stats is not None:
            for col in LLM_METRIC_NAMES:
                mean_row[col] = ''
                ci_row[col] = ''

        # Paired-test p values and significance markers.
        for metric, p_col, star_col in zip(ALL_SIG_METRICS, SIG_P_COLS, SIG_STAR_COLS):
            if sig_pvals is not None and method_name in sig_pvals:
                p = sig_pvals[method_name].get(metric, float('nan'))
                if np.isnan(p):
                    mean_row[p_col] = '-'
                    mean_row[star_col] = '-'
                else:
                    mean_row[p_col] = f'{p:.4e}'
                    mean_row[star_col] = p_to_stars(p)
            else:
                mean_row[p_col] = ''
                mean_row[star_col] = ''

            ci_row[p_col] = ''
            ci_row[star_col] = ''

        records.append(mean_row)
        records.append(ci_row)

    df = pd.DataFrame(records, columns=all_cols)
    return df


# Center-level dataset aggregation
def build_center_combined(nlp_raw, llm_raw_counts):
    result = {}

    for center_label, sub_datasets in center_groups.items():
        print(f'\n  合并 {center_label}: {sub_datasets}')

        nlp_stats = []
        merged_nlp_raw = {}
        for method_name in method_names:
            frames = []
            for ds in sub_datasets:
                if ds in nlp_raw and method_name in nlp_raw[ds]:
                    df_part = nlp_raw[ds][method_name]
                    if not df_part.empty:
                        frames.append(df_part)
            if frames:
                merged = pd.concat(frames, ignore_index=True)
                merged_nlp_raw[method_name] = merged
                means, cis = [], []
                for col in NLP_METRIC_NAMES:
                    mean, lo, hi = bootstrap_nlp_ci(merged[col].values)
                    means.append(mean)
                    cis.append((lo, hi))
            else:
                merged_nlp_raw[method_name] = pd.DataFrame(columns=NLP_METRIC_NAMES)
                means = [float('nan')] * len(NLP_METRIC_NAMES)
                cis = [(float('nan'), float('nan'))] * len(NLP_METRIC_NAMES)
            nlp_stats.append((method_name, means, cis))

        llm_stats = None
        merged_llm_raw = {}
        if llm_raw_counts is not None:
            llm_stats = []
            for method_name in method_names:
                combined = []
                for ds in sub_datasets:
                    if ds in llm_raw_counts and method_name in llm_raw_counts[ds]:
                        combined.extend(llm_raw_counts[ds][method_name])
                merged_llm_raw[method_name] = combined
                if combined:
                    tps = [x[0] for x in combined]
                    fps = [x[1] for x in combined]
                    fns = [x[2] for x in combined]
                    total_tp, total_fp, total_fn = sum(tps), sum(fps), sum(fns)
                    (p_mean, p_lo, p_hi), (r_mean, r_lo, r_hi), (f1_mean, f1_lo, f1_hi) = \
                        bootstrap_prf_ci(tps, fps, fns)
                    llm_v = {'TP': total_tp, 'FP': total_fp, 'FN': total_fn}
                    llm_c = [(p_mean, p_lo, p_hi), (r_mean, r_lo, r_hi), (f1_mean, f1_lo, f1_hi)]
                    llm_stats.append((method_name, llm_v, llm_c))
                else:
                    llm_stats.append((method_name, None, None))

        print(f'\n  计算显著性检验 ({center_label})...')
        sig_pvals = compute_significance_columns(
            merged_nlp_raw, merged_llm_raw if llm_raw_counts is not None else None)

        df = build_two_row_df(nlp_stats, llm_stats, sig_pvals)
        result[center_label] = df

    return result


# CSV output
def save_combined_csv(nlp_stats_dict, llm_stats_dict, nlp_raw, llm_raw_counts):
    for base_name in base_dataset_list:
        if base_name not in nlp_stats_dict:
            continue
        nlp_stats = nlp_stats_dict[base_name]
        llm_stats = llm_stats_dict[base_name] if llm_stats_dict else None

        nlp_raw_for_ds = nlp_raw.get(base_name, {})
        llm_raw_for_ds = llm_raw_counts.get(base_name, {}) if llm_raw_counts else None
        print(f'\n  计算显著性检验 ({base_name})...')
        sig_pvals = compute_significance_columns(nlp_raw_for_ds, llm_raw_for_ds)

        df = build_two_row_df(nlp_stats, llm_stats, sig_pvals)
        out_csv = os.path.join(
            output_combined_dir, f'[Combined]_{dataset_to_csv_name(base_name)}.csv')
        df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f'  已保存: {out_csv}')


def save_center_csv(center_dict):
    for center_label, df in center_dict.items():
        out_csv = os.path.join(output_combined_dir, f'[Combined]_{center_label}.csv')
        df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f'  已保存: {out_csv}')


# Main analysis pipeline
if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('统计汇总开始（单侧检验: H1: FetalScribe > other）')
    print('=' * 60)

    print('\n--- Part A: NLP 指标 ---')
    nlp_stats_dict, nlp_raw = load_nlp_stats()

    print('\n--- Part B: LLM 指标 ---')
    llm_stats_dict, llm_raw_counts = load_llm_stats()

    print('\n--- 保存逐数据集 Combined CSV ---')
    save_combined_csv(nlp_stats_dict, llm_stats_dict, nlp_raw, llm_raw_counts)

    print('\n--- 按 Center 合并 ---')
    center_dict = build_center_combined(nlp_raw, llm_raw_counts)
    save_center_csv(center_dict)

    print('\n全部统计完成！')
