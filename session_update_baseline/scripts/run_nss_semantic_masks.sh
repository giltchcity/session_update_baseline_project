#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/jixian/Desktop/miniconda3/envs/3d_vsg/bin/python"
MODEL_DIR="/home/jixian/.cache/huggingface/hub/models--nvidia--segformer-b5-finetuned-ade-640-640/snapshots/739f5d4692954e4a185eac280dec1ba5a7d52f1d"

export PYTHONPATH="${ROOT}/../envs/nss-semantics-py38${PYTHONPATH:+:${PYTHONPATH}}"
export TRANSFORMERS_CACHE="${ROOT}/../envs/hf-cache"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

exec "${PYTHON}" "${ROOT}/scripts/generate_nss_ade20k_masks.py" \
  --model-dir "${MODEL_DIR}" "$@"
