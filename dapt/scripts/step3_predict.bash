#!/bin/bash
# DAPT prediction script
# Usage: CUDA_VISIBLE_DEVICES=0 bash step3_predict.bash [VERSION] [MODE]
#   VERSION: v1 | v2_kg | all (default: all)
#   MODE: sft_only | sft_dapt | all (default: all)
#
# Example: bash step3_predict.bash v1 sft_only

set -e

VERSION="${1:-all}"
MODE_CHOICE="${2:-all}"

DAPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${DAPT_DIR}"

MODEL_PATH="/path/to/DeepSeek-R1-0528-Qwen3-8B"

# Dataset directories for each version
declare -A DATA_DIRS
DATA_DIRS["v1"]="/path/to/direct_finetuning/data"
DATA_DIRS["v2_kg"]="/path/to/kg_enhanced_finetuning/data"

# Test datasets
DATASETS=(
    "InDistribution_test"
    "OutofDistribution_center1_normal"
    "OutofDistribution_center1_patients"
    "OutofDistribution_center2_normal"
    "OutofDistribution_center2_patients"
)

# SFT adapter paths
declare -A SFT_ADAPTERS
SFT_ADAPTERS["v1"]="/path/to/direct_finetuning/saves/fetal_mri_lora"
SFT_ADAPTERS["v2_kg"]="/path/to/kg_enhanced_finetuning/saves/fetal_mri_plus_kg_lora"

# Determine versions to run
if [ "${VERSION}" = "all" ]; then
    VERSIONS=("v1" "v2_kg")
else
    VERSIONS=("${VERSION}")
fi

# Determine modes to run
if [ "${MODE_CHOICE}" = "all" ]; then
    MODES_TO_RUN=("sft_only" "sft_dapt")
elif [ "${MODE_CHOICE}" = "sft_only" ] || [ "${MODE_CHOICE}" = "sft_dapt" ]; then
    MODES_TO_RUN=("${MODE_CHOICE}")
else
    echo "Unknown mode: ${MODE_CHOICE}, options: sft_only sft_dapt all"
    exit 1
fi

run_predict_for_version() {
    local VER="$1"
    local SFT_PATH="${SFT_ADAPTERS[$VER]}"
    local EVAL_DATA_DIR="${DATA_DIRS[$VER]}"

    echo ""
    echo "============================================================"
    echo "  Version: ${VER}  SFT adapter: ${SFT_PATH}"
    echo "  Dataset dir: ${EVAL_DATA_DIR}"
    echo "============================================================"

    for MODE in "${MODES_TO_RUN[@]}"; do
        echo ""
        echo "  >>> Mode: ${VER} | ${MODE}"

        for DATASET in "${DATASETS[@]}"; do

            if [ "${MODE}" = "sft_only" ]; then
                ADAPTER_PATH="${SFT_PATH}"
            else
                if [[ "${DATASET}" == *"center1"* ]]; then
                    DAPT_PATH="${DAPT_DIR}/saves/${VER}/center1_dapt"
                    ADAPTER_PATH="${SFT_PATH},${DAPT_PATH}"
                elif [[ "${DATASET}" == *"center2"* ]]; then
                    DAPT_PATH="${DAPT_DIR}/saves/${VER}/center2_dapt"
                    ADAPTER_PATH="${SFT_PATH},${DAPT_PATH}"
                else
                    echo "    --- Skip: ${DATASET} (no DAPT needed for in-distribution)"
                    continue
                fi
            fi

            OUTPUT_DIR="${DAPT_DIR}/predictions/${VER}/${MODE}/${DATASET}"
            mkdir -p "${OUTPUT_DIR}"

            TEMP_YAML="${DAPT_DIR}/configs/tmp_predict_${VER}_${MODE}_${DATASET}.yaml"

            cat > "${TEMP_YAML}" <<YAML
### ===== Model =====
model_name_or_path: ${MODEL_PATH}
adapter_name_or_path: ${ADAPTER_PATH}
trust_remote_code: true

### ===== Task =====
stage: sft
do_train: false
do_eval: false
do_predict: true
finetuning_type: lora
template: deepseekr1

### ===== Thinking Chain =====
enable_thinking: false

### ===== Dataset =====
dataset_dir: ${EVAL_DATA_DIR}
eval_dataset: ${DATASET}
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### ===== Generation Parameters =====
predict_with_generate: true
per_device_eval_batch_size: 1
ddp_timeout: 180000000
bf16: true
temperature: 0.2
top_p: 0.2
max_new_tokens: 2048

### ===== Output =====
output_dir: ${OUTPUT_DIR}
overwrite_output_dir: true
report_to: none
YAML

            echo "    Predicting: ${DATASET} -> ${OUTPUT_DIR}"
            llamafactory-cli train "${TEMP_YAML}"

            if [ $? -eq 0 ]; then
                echo "    ${DATASET} done"
                rm -f "${TEMP_YAML}"
            else
                echo "    ${DATASET} failed! Config kept: ${TEMP_YAML}"
                exit 1
            fi
        done
    done
}

for VER in "${VERSIONS[@]}"; do
    if [ -z "${SFT_ADAPTERS[$VER]}" ]; then
        echo "Unknown version: ${VER}, options: v1 v2_kg all"
        exit 1
    fi
    run_predict_for_version "${VER}"
done

echo ""
echo "============================================================"
echo "  All predictions completed! Results: ${DAPT_DIR}/predictions/"
echo "============================================================"
