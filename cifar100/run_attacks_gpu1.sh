#!/usr/bin/env bash
set -euo pipefail

PYTHON="/mnt/sdz/abka03_data/env/xaibackdoors/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ATTACK_SCRIPT="${REPO_ROOT}/attack.py"

run_attack() {
    local model_id="$1"

    echo "=== Running ${model_id} on cuda:1 ==="
    "${PYTHON}" "${ATTACK_SCRIPT}" "cuda:1" "${model_id}" --grad-layer -1
}

main() {
    local models=(CIFAR100VGG154L1 CIFAR100MNT154L1 CIFAR100RNT154L1)

    for model in "${models[@]}"; do
        run_attack "${model}"
    done
}

main "$@"
