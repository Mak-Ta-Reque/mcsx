#!/usr/bin/env bash
set -euo pipefail

PYTHON="/mnt/sdz/abka03_data/env/xaibackdoors/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SCRIPT="${REPO_ROOT}/evaluate_attack_defense_metrics.py"
SAVE_PATH="${REPO_ROOT}/output/cifar100"

run_evaluation() {
    local model_id="$1"

    echo "=== Evaluating ${model_id} on cuda:0 ==="
    "${PYTHON}" "${EVAL_SCRIPT}" "${model_id}" \
        --device "cuda:0" \
        --datasize "10000" \
        --batchsize "128" \
        --save_path "${SAVE_PATH}"
}

main() {
    local models=(CIFAR100VIT154L1 CIFAR100WRN154L1)

    for model in "${models[@]}"; do
        run_evaluation "${model}"
    done
}

main "$@"
