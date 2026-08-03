#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/03_train_dapt.sh [direct|kg|all]
BRANCH="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAPT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DAPT_DIR}"

case "${BRANCH}" in
  direct) CONFIGS=(direct_center1 direct_center2) ;;
  kg) CONFIGS=(kg_center1 kg_center2) ;;
  all) CONFIGS=(direct_center1 direct_center2 kg_center1 kg_center2) ;;
  *)
    echo "Unknown branch: ${BRANCH}. Expected direct, kg, or all." >&2
    exit 2
    ;;
esac

for config in "${CONFIGS[@]}"; do
  echo "Training DAPT adapter with configs/${config}.yaml"
  llamafactory-cli train "configs/${config}.yaml"
done
