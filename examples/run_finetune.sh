#!/bin/bash

# This script demonstrates how to fine-tune OmniVoice from a JSONL manifest.

set -euo pipefail

stage=0
stop_stage=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ====== Modify as needed ======
# GPUs to use
GPU_IDS="0,1"
NUM_GPUS=2

# Path to your input JSONL file
# (each line: {"id": ..., "audio_path": ..., "text": ..., "language_id": ...})
TRAIN_JSONL="datasets/ngoc_huyen_vbee_audiobook/train_raw.jsonl"

# Path to your dev JSONL file. Set to empty string to skip dev set.
DEV_JSONL=

# Directory to write tokenized WebDataset shards
TOKEN_DIR="data/finetune/tokens"

# Audio tokenizer model (HuggingFace repo or local path)
TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"

# Training config file
# If you encounter issues with flex_attention on your GPU, use the SDPA config instead:
# TRAIN_CONFIG="examples/config/train_config_finetune_sdpa.json"
TRAIN_CONFIG="examples/config/train_config_finetune.json"

# Optional explicit python binary. Leave empty to auto-detect.
PYTHON_BIN="/Users/leakbyte/workspace/codex2/tts-cli/omnivoice-venv/bin/python"

# Runtime data config generated from available manifests.
DATA_CONFIG_RUNTIME="${TOKEN_DIR}/data_config.runtime.json"

# Output directory for fine-tuned checkpoints
OUTPUT_DIR="exp/omnivoice_finetune"
# =================================

if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3)"
fi

if [ -z "${PYTHON_BIN}" ]; then
    echo "Error: Cannot find Python executable. Set PYTHON_BIN in this script."
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"


# Stage 0: Tokenize audio into WebDataset shards
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "Stage 0: Tokenizing audio"

    for split_jsonl_path in ${TRAIN_JSONL} ${DEV_JSONL}; do
        if [ -z "${split_jsonl_path}" ]; then
            continue
        fi

        if [ "${split_jsonl_path}" = "${TRAIN_JSONL}" ]; then
            split="train"
        else
            split="dev"
        fi

        echo "  Tokenizing ${split} from ${split_jsonl_path}"

        CUDA_VISIBLE_DEVICES=${GPU_IDS} \
            "${PYTHON_BIN}" -m omnivoice.scripts.extract_audio_tokens \
            --input_jsonl "${split_jsonl_path}" \
            --tar_output_pattern "${TOKEN_DIR}/${split}/audios/shard-%06d.tar" \
            --jsonl_output_pattern "${TOKEN_DIR}/${split}/txts/shard-%06d.jsonl" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --nj_per_gpu 3 \
            --loader_workers 8 \
            --shuffle True

        echo "  Done. Manifest written to ${TOKEN_DIR}/${split}/data.lst"
    done
fi


# Build runtime data config according to available manifests.
mkdir -p "$(dirname "${DATA_CONFIG_RUNTIME}")"
if [ -n "${DEV_JSONL:-}" ]; then
        cat > "${DATA_CONFIG_RUNTIME}" <<EOF
{
    "train": [
        { "manifest_path": ["${TOKEN_DIR}/train/data.lst"] }
    ],
    "dev": [
        { "manifest_path": ["${TOKEN_DIR}/dev/data.lst"] }
    ]
}
EOF
else
        cat > "${DATA_CONFIG_RUNTIME}" <<EOF
{
    "train": [
        { "manifest_path": ["${TOKEN_DIR}/train/data.lst"] }
    ]
}
EOF
fi

if [ ! -f "${TOKEN_DIR}/train/data.lst" ]; then
        echo "Error: missing ${TOKEN_DIR}/train/data.lst. Run stage 0 first."
        exit 1
fi


# Stage 1: Fine-tune
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "Stage 1: Fine-tuning"

        "${PYTHON_BIN}" -m accelerate.commands.launch \
        --gpu_ids "${GPU_IDS}" \
        --num_processes ${NUM_GPUS} \
        -m omnivoice.cli.train \
        --train_config ${TRAIN_CONFIG} \
                --data_config ${DATA_CONFIG_RUNTIME} \
        --output_dir ${OUTPUT_DIR}
fi
