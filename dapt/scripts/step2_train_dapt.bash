#!/bin/bash
# DAPT batch training script
# Usage: CUDA_VISIBLE_DEVICES=0 bash step2_train_dapt.bash

set -e

DAPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${DAPT_DIR}"

CONFIGS=(
    "v1_center1_dapt"
    "v1_center2_dapt"
    "v2_kg_center1_dapt"
    "v2_kg_center2_dapt"
)

TOTAL=${#CONFIGS[@]}
COUNT=0

echo ""
echo "============================================================"
echo "  DAPT batch training: ${TOTAL} configs"
echo "============================================================"

for CFG in "${CONFIGS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "============================================================"
    echo "  [${COUNT}/${TOTAL}] Training: ${CFG}"
    echo "============================================================"

    llamafactory-cli train "${DAPT_DIR}/configs/${CFG}.yaml"

    if [ $? -eq 0 ]; then
        echo "  ${CFG} completed!"
    else
        echo "  ${CFG} failed!"
        exit 1
    fi
done

echo ""
echo "============================================================"
echo "  All ${TOTAL} DAPT trainings completed!"
echo "============================================================"
