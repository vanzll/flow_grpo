#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-config/prior_dit_offline.py:pickscore_sd3_dit_offline_8gpu_h20}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29511}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/accelerate_configs/multi_gpu.yaml}"
DEFAULT_ENV_PATH="${HOME}/.conda/envs/flow_grpo"
ACCELERATE_BIN="${ACCELERATE_BIN:-}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

if [[ -z "${ACCELERATE_BIN}" ]]; then
  if [[ -x "${DEFAULT_ENV_PATH}/bin/accelerate" ]]; then
    ACCELERATE_BIN="${DEFAULT_ENV_PATH}/bin/accelerate"
  else
    ACCELERATE_BIN="$(command -v accelerate || true)"
  fi
fi

if [[ -z "${ACCELERATE_BIN}" || ! -x "${ACCELERATE_BIN}" ]]; then
  echo "accelerate not found. Set ACCELERATE_BIN or install/activate the flow_grpo env." >&2
  exit 1
fi

"${ACCELERATE_BIN}" launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  scripts/train_prior_dit_offline.py \
  --config "${CONFIG}" \
  "$@"
