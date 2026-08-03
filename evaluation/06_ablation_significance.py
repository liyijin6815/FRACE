"""Run paired significance tests for the combined cross-center ablation set.

The directional alternative is FetalScribe > comparison method. NLP metrics
use one-sided Wilcoxon signed-rank tests; clinical precision, recall, and F1
use one-sided paired permutation tests.
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

FETALSCRIBE_NAME = 'FetalScribe'

# Ablation methods included in the manuscript comparison.
file_path_list = [
    'finetuning/deepseek-r1-qwen3-8b',
    'finetuning_sr/deepseek-r1-qwen3-8b',
    'finetuning_kg_v3/deepseek-r1-qwen3-8b',
    'finetuning_DAPT/deepseek-r1-qwen3-8b',
    'finetuning_ICL/deepseek-r1-qwen3-8b',
    'finetuning_combined_ICL_ICL/deepseek-r1-qwen3-8b',
    'FetalScribe/deepseek-r1-qwen3-8b',
]

method_names = [
    'fine-tuned',
    'fine-tuned-sr',
    'fine-tuned-kg_v3',
    'fine-tuned_DAPT',
    'fine-tuned_ICL',
    'fine-tuned_combined_ICL',
    'FetalScribe',
]

# Four subsets combined for the cross-center OoD analysis.
ood_dataset_list = [
    'OutofDistribution_center1_normal.json',
    'OutofDistribution_center1_patients.json',
    'OutofDistribution_center2_normal.json',
    'OutofDistribution_center2_patients.json',
]

assert len(file_path_list) == len(method_names)

NLP_METRIC_NAMES = ['rouge_p', 'rouge_r', 'rouge_f1',
                    'bert_p', 'bert_r', 'bert_f1', 'sentence_sim']

# Metrics included in paired significance tests.
SIG_NLP_METRICS = ['rouge_p', 'rouge_r', 'rouge_f1',
                   'bert_p', 'bert_r', 'bert_f1', 'sentence_sim']
SIG_LLM_METRICS = ['Precision', 'Recall', 'F1']


# Metric and formatting utilities
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


def bootstrap_ci(values, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
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
    total_tp, total_fp, total_fn = arr[:, 0].sum(), arr[:, 1].sum(), arr[:, 2].sum()
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
    arr_a, arr_b = arr_a[valid], arr_b[valid]
    if len(arr_a) < 5:
        return float('nan')
    diff = arr_a - arr_b
    if np.all(diff == 0):
        return 1.0
    try:
        # ``greater`` tests whether arr_a tends to exceed arr_b.
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


# Metric loading
def load_and_merge_data():
    merged_nlp = {m: [] for m in method_names}
    merged_llm = {m: [] for m in method_names}

    for ds_name in ood_dataset_list:
        ds_stem = ds_name.replace('.json', '')
        print(f'\n  加载 {ds_name}...')

        for method_idx, method_name in enumerate(method_names):
            model_path = file_path_list[method_idx]

            # --- NLP ---
            csv_path = os.path.join(scores_root_path, model_path,
                                    f'[Scores]_{ds_stem}.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)[NLP_METRIC_NAMES]
                merged_nlp[method_name].append(df)
                print(f'    {method_name}: NLP {len(df)} 条')
            else:
                print(f'    警告: {csv_path} 不存在')

            # --- LLM ---
            json_path = os.path.join(LLM_root_path, model_path,
                                     f'[LLM_Eva]_{ds_stem}.json')
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    text = item.get('model_result', '')
                    tp, fp, fn = parse_tp_fp_fn(text)
                    merged_llm[method_name].append((tp, fp, fn))
                print(f'    {method_name}: LLM {len(data)} 条')
            else:
                print(f'    警告: {json_path} 不存在')

    for m in method_names:
        if merged_nlp[m]:
            merged_nlp[m] = pd.concat(merged_nlp[m], ignore_index=True)
        else:
            merged_nlp[m] = pd.DataFrame(columns=NLP_METRIC_NAMES)

    return merged_nlp, merged_llm


# Main ablation analysis
def main():
    print('=' * 60)
    print('消融实验显著性分析: OOD1 + OOD2 合并 (单侧检验)')
    print('=' * 60)

    # Load all available method results.
    merged_nlp, merged_llm = load_and_merge_data()

    # Report aligned sample counts.
    print('\n--- 合并后样本量 ---')
    for m in method_names:
        n_nlp = len(merged_nlp[m]) if not merged_nlp[m].empty else 0
        n_llm = len(merged_llm[m])
        print(f'  {m}: NLP={n_nlp}, LLM={n_llm}')

    # Use FetalScribe as the reference method.
    fs_nlp = merged_nlp[FETALSCRIBE_NAME]
    fs_llm = merged_llm[FETALSCRIBE_NAME]

    # Compute point estimates, intervals, and paired tests.
    all_metrics = SIG_NLP_METRICS + SIG_LLM_METRICS
    rows = []

    for method_name in method_names:
        print(f'\n--- {method_name} ---')

        m_nlp = merged_nlp[method_name]
        m_llm = merged_llm[method_name]

        # NLP metrics
        row_mean = {'Method': method_name}
        row_ci = {'Method': ''}

        for col in SIG_NLP_METRICS:
            if not m_nlp.empty and col in m_nlp.columns:
                mean, lo, hi = bootstrap_ci(m_nlp[col].values)
            else:
                mean, lo, hi = float('nan'), float('nan'), float('nan')

            row_mean[col] = round(mean, 3) if not np.isnan(mean) else ''
            row_ci[col] = fmt_ci(lo, hi)

        # LLM-based clinical metrics
        if m_llm:
            tps = [x[0] for x in m_llm]
            fps = [x[1] for x in m_llm]
            fns = [x[2] for x in m_llm]
            total_tp, total_fp, total_fn = sum(tps), sum(fps), sum(fns)

            (p_m, p_lo, p_hi), (r_m, r_lo, r_hi), (f1_m, f1_lo, f1_hi) = \
                bootstrap_prf_ci(tps, fps, fns)

            row_mean['TP'] = total_tp
            row_mean['FP'] = total_fp
            row_mean['FN'] = total_fn
            row_mean['Precision'] = round(p_m, 3)
            row_mean['Recall'] = round(r_m, 3)
            row_mean['F1'] = round(f1_m, 3)

            row_ci['TP'] = ''
            row_ci['FP'] = ''
            row_ci['FN'] = ''
            row_ci['Precision'] = fmt_ci(p_lo, p_hi)
            row_ci['Recall'] = fmt_ci(r_lo, r_hi)
            row_ci['F1'] = fmt_ci(f1_lo, f1_hi)
        else:
            for c in ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']:
                row_mean[c] = ''
                row_ci[c] = ''

        # Paired significance tests
        if method_name == FETALSCRIBE_NAME:
            for metric in all_metrics:
                row_mean[f'{metric}_sig'] = '-'
                row_ci[f'{metric}_sig'] = ''
        else:
            # NLP metrics: one-sided paired Wilcoxon test.
            for col in SIG_NLP_METRICS:
                fs_arr = fs_nlp[col].values if not fs_nlp.empty and col in fs_nlp.columns else np.array([])
                m_arr = m_nlp[col].values if not m_nlp.empty and col in m_nlp.columns else np.array([])

                if len(fs_arr) > 0 and len(m_arr) > 0 and len(fs_arr) == len(m_arr):
                    p_val = paired_wilcoxon_test_onesided(fs_arr, m_arr)
                else:
                    p_val = float('nan')

                row_mean[f'{col}_sig'] = p_to_stars(p_val)
                row_ci[f'{col}_sig'] = ''

                print(f'  Wilcoxon(1-sided) {col}: p={p_val:.6e}  {p_to_stars(p_val)}')

            # Clinical metrics: one-sided paired permutation test.
            for metric in SIG_LLM_METRICS:
                if fs_llm and m_llm and len(fs_llm) == len(m_llm):
                    p_val = paired_permutation_test_micro_onesided(fs_llm, m_llm, metric=metric)
                else:
                    p_val = float('nan')

                row_mean[f'{metric}_sig'] = p_to_stars(p_val)
                row_ci[f'{metric}_sig'] = ''

                print(f'  Permutation(1-sided) {metric}: p={p_val:.6e}  {p_to_stars(p_val)}')

        rows.append(row_mean)
        rows.append(row_ci)

    # Build the publication table.
    cols = ['Method']
    cols += SIG_NLP_METRICS
    cols += ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
    for metric in all_metrics:
        cols += [f'{metric}_sig']

    df = pd.DataFrame(rows, columns=cols)

    # Save machine-readable results.
    out_path = os.path.join(output_combined_dir, '[Ablation_Significance]_OOD_all.csv')
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'\n已保存: {out_path}')

    # Print a compact summary.
    print('\n' + '=' * 80)
    print('显著性汇总 (vs FetalScribe, one-sided: FetalScribe > other)')
    print('=' * 80)

    header = f'{"Method":<30}'
    for metric in all_metrics:
        header += f'{metric:>14}'
    print(header)
    print('-' * (30 + 14 * len(all_metrics)))

    for i in range(0, len(rows), 2):
        row = rows[i]
        method = row['Method']
        line = f'{method:<30}'
        for metric in all_metrics:
            sig = row.get(f'{metric}_sig', '')
            line += f'{sig:>14}'
        print(line)

    print('\n* p<0.05, ** p<0.01, *** p<0.001, n.s. = not significant')
    print('H1: FetalScribe > ablated variant (one-sided)')
    print('NLP metrics: Wilcoxon signed-rank test (one-sided, alternative="greater")')
    print('LLM metrics: Paired permutation test (one-sided)')
    print(f'Permutation iterations: {PERMUTATION_N}')


if __name__ == '__main__':
    main()
