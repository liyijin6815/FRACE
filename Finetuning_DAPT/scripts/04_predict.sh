#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/04_predict.sh [direct|kg|all] [sft_only|sft_dapt|all]
BRANCH_CHOICE="${1:-all}"
MODE_CHOICE="${2:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAPT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${DAPT_DIR}/.." && pwd)"
BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-R1-0528-Qwen3-8B}"
DAPT_ADAPTER_ROOT="${DAPT_ADAPTER_ROOT:-${DAPT_DIR}/adapters}"

DATASETS=(
  InDistribution_test
  OutofDistribution_center1_normal
  OutofDistribution_center1_patients
  OutofDistribution_center2_normal
  OutofDistribution_center2_patients
)

case "${BRANCH_CHOICE}" in
  direct) BRANCHES=(direct) ;;
  kg) BRANCHES=(kg) ;;
  all) BRANCHES=(direct kg) ;;
  *) echo "Unknown branch: ${BRANCH_CHOICE}" >&2; exit 2 ;;
esac

case "${MODE_CHOICE}" in
  sft_only) MODES=(sft_only) ;;
  sft_dapt) MODES=(sft_dapt) ;;
  all) MODES=(sft_only sft_dapt) ;;
  *) echo "Unknown mode: ${MODE_CHOICE}" >&2; exit 2 ;;
esac

for branch in "${BRANCHES[@]}"; do
  if [[ "${branch}" == "direct" ]]; then
    SFT_ADAPTER="${ROOT_DIR}/Finetuning/adapter/fetal_mri_lora"
    DATA_DIR="${ROOT_DIR}/Finetuning/data"
  else
    SFT_ADAPTER="${ROOT_DIR}/Finetuning_KG/adapter/fetal_mri_plus_kg_lora_v3"
    DATA_DIR="${ROOT_DIR}/Finetuning_KG/data"
  fi

  for mode in "${MODES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      ADAPTERS="${SFT_ADAPTER}"
      if [[ "${mode}" == "sft_dapt" ]]; then
        if [[ "${dataset}" == *center1* ]]; then
          ADAPTERS="${SFT_ADAPTER},${DAPT_ADAPTER_ROOT}/${branch}/center1"
        elif [[ "${dataset}" == *center2* ]]; then
          ADAPTERS="${SFT_ADAPTER},${DAPT_ADAPTER_ROOT}/${branch}/center2"
        else
          continue
        fi
      fi

      OUTPUT_DIR="${DAPT_DIR}/predictions/${branch}/${mode}/${dataset}"
      TEMP_CONFIG="${DAPT_DIR}/configs/.tmp_${branch}_${mode}_${dataset}.yaml"
      mkdir -p "${OUTPUT_DIR}"

      cat > "${TEMP_CONFIG}" <<YAML
model_name_or_path: ${BASE_MODEL}
adapter_name_or_path: ${ADAPTERS}
trust_remote_code: true
stage: sft
do_train: false
do_eval: false
do_predict: true
finetuning_type: lora
template: deepseekr1
enable_thinking: false
dataset_dir: ${DATA_DIR}
eval_dataset: ${dataset}
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4
predict_with_generate: true
per_device_eval_batch_size: 1
bf16: true
temperature: 0.2
top_p: 0.2
max_new_tokens: 2048
ddp_timeout: 180000000
output_dir: ${OUTPUT_DIR}
overwrite_output_dir: true
report_to: none
YAML

      echo "Predicting ${branch}/${mode}/${dataset}"
      llamafactory-cli train "${TEMP_CONFIG}"
      rm -f "${TEMP_CONFIG}"
    done
  done
done
