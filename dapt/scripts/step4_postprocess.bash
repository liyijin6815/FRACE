#!/bin/bash
# Post-prediction evaluation pipeline
# Usage: CUDA_VISIBLE_DEVICES=0 bash step4_postprocess.bash [VERSION] [MODE]
# Example: bash step4_postprocess.bash v2_kg sft_dapt
#          bash step4_postprocess.bash v1 sft_dapt
#          (no args = process all versions and modes)

set -e

DAPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="${DAPT_DIR}/scripts"

VERSION_FILTER="${1:-}"
MODE_FILTER="${2:-}"

VERSIONS=("v1" "v2_kg")
MODES=("sft_only" "sft_dapt")

DATASETS=(
    "InDistribution_test"
    "OutofDistribution_center1_normal"
    "OutofDistribution_center1_patients"
    "OutofDistribution_center2_normal"
    "OutofDistribution_center2_patients"
)

echo ""
echo "============================================================"
echo "  Post-prediction evaluation pipeline"
echo "  Results saved to: ${DAPT_DIR}/results/"
echo "============================================================"

for VERSION in "${VERSIONS[@]}"; do
    [[ -n "$VERSION_FILTER" && "$VERSION" != "$VERSION_FILTER" ]] && continue

    for MODE in "${MODES[@]}"; do
        [[ -n "$MODE_FILTER" && "$MODE" != "$MODE_FILTER" ]] && continue

        PRED_DIR="${DAPT_DIR}/predictions/${VERSION}/${MODE}"
        RESULT_DIR="${DAPT_DIR}/results/${VERSION}/${MODE}"

        if [ ! -d "${PRED_DIR}" ]; then
            echo "  Prediction dir not found, skip: ${PRED_DIR}"
            continue
        fi

        FOUND=0
        for DATASET in "${DATASETS[@]}"; do
            if [ -d "${PRED_DIR}/${DATASET}" ]; then
                FOUND=$((FOUND + 1))
            fi
        done
        if [ "$FOUND" -eq 0 ]; then
            echo "  ${VERSION}/${MODE}: No predictions found, run step3 first"
            continue
        fi

        echo ""
        echo ">>> Processing: ${VERSION} / ${MODE}  (${FOUND} datasets found)"
        echo "------------------------------------------------------------"

        DIAG_ROOT="${RESULT_DIR}/test_results_diagnosis"
        SCORES_ROOT="${RESULT_DIR}/test_results_scores"
        LLM_ROOT="${RESULT_DIR}/test_results_LLM"
        METRICS_ROOT="${RESULT_DIR}/test_results_metrics"

        mkdir -p "${DIAG_ROOT}" "${SCORES_ROOT}" "${LLM_ROOT}" "${METRICS_ROOT}"

        # Step 1: Text extraction and cleaning
        echo "  Step 1: Text extraction and cleaning"
        PRED_DIR="${PRED_DIR}" \
        DIAG_DIR="${DIAG_ROOT}/model" \
            python "${SCRIPTS_DIR}/step4-1_text_extract_and_clean.py"

        # Step 2: NLP metrics (ROUGE-L / BERTScore / SentSim)
        echo "  Step 2: NLP metrics evaluation"
        DIAG_ROOT="${DIAG_ROOT}" \
        SCORES_ROOT="${SCORES_ROOT}" \
        MODEL_NAME="model" \
            python "${SCRIPTS_DIR}/step4-2_evaluate_output_with_metrics.py"

        # Step 3 Part 1: LLM keyword extraction
        echo "  Step 3 Part 1: LLM keyword extraction"
        DIAG_ROOT="${DIAG_ROOT}" \
        LLM_ROOT="${LLM_ROOT}" \
        MODEL_NAME="model" \
            python "${SCRIPTS_DIR}/step4-3_evaluate_outputs_with_llm_part1.py"

        # Step 3 Part 2: LLM TP/FP/FN comparison
        echo "  Step 3 Part 2: LLM TP/FP/FN comparison"
        LLM_ROOT="${LLM_ROOT}" \
        MODEL_NAME="model" \
            python "${SCRIPTS_DIR}/step4-3_evaluate_outputs_with_llm_part2.py"

        # Step 4: Aggregate statistics
        echo "  Step 4: Aggregate statistics"
        SCORES_ROOT="${SCORES_ROOT}" \
        LLM_ROOT="${LLM_ROOT}" \
        METRICS_ROOT="${METRICS_ROOT}" \
        MODEL_NAME="model" \
            python "${SCRIPTS_DIR}/step4-4_get_statistics_with_metrics.py"

        echo "  ${VERSION}/${MODE} evaluation completed"
        echo "     Results: ${METRICS_ROOT}"
    done
done

echo ""
echo "============================================================"
echo "  All evaluations completed!"
echo "  Results directory: ${DAPT_DIR}/results/"
echo "============================================================"
