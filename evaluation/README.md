# Evaluation pipeline

This directory contains the post-processing and evaluation implementation used in the manuscript.

## Stages

1. `01_clean_predictions.py` reads LLaMA-Factory `generated_predictions.jsonl` files and writes normalized JSON files.
2. `02_compute_nlp_metrics.py` computes per-record ROUGE-L, BERTScore, and sentence similarity.
3. `03_extract_clinical_keywords.py` uses GPT-5.2 to extract structured abnormalities from predictions and references.
4. `04_match_clinical_findings.py` uses GPT-5.2 to semantically match the extracted abnormalities and count TP, FP, and FN.
5. `05_statistical_analysis.py` computes 95% bootstrap confidence intervals and paired significance tests.
6. `06_ablation_significance.py` performs the combined cross-center ablation significance analysis.

## Expected layout

Stage 1 reads:

```text
${PREDICTIONS_ROOT}/<dataset>/generated_predictions.jsonl
```

For each model, set `CLEANED_ROOT` to that model's destination under the shared result tree. Stages 2-6 expect:

```text
test_results_diagnosis/<model>/[Cleaned]_<dataset>.json
test_results_scores/<model>/[Scores]_<dataset>.csv
test_results_LLM/<model>/<dataset>.json
test_results_LLM/<model>/[LLM_Eva]_<dataset>.json
```

The model directory names are defined by `file_path_list` near the top of stages 2-6. Missing model or dataset files are skipped so that a subset can be evaluated without changing the algorithms.

## Environment

All paths are relative by default and can be overridden:

```bash
export PREDICTIONS_ROOT=./Finetuning/predictions
export CLEANED_ROOT=./test_results_diagnosis/finetuning/deepseek-r1-qwen3-8b
export SCORES_ROOT=./test_results_scores
export LLM_ROOT=./test_results_LLM
export METRICS_ROOT=./test_results_metrics
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export EVAL_MODEL="gpt-5.2"
export EVAL_MAX_WORKERS=6
```

Run stage 1 once for each model prediction directory. Then set `CLEANED_ROOT=./test_results_diagnosis` and run the remaining stages:

```bash
python evaluation/01_clean_predictions.py
python evaluation/02_compute_nlp_metrics.py
python evaluation/03_extract_clinical_keywords.py
python evaluation/04_match_clinical_findings.py
python evaluation/05_statistical_analysis.py
python evaluation/06_ablation_significance.py
```

The statistical settings are fixed in source for manuscript reproducibility: 1,000 bootstrap resamples, random seed 42, and 10,000 paired permutations.
