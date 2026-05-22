#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--smoke" || "${SMOKE:-0}" == "1" ]]; then
  EPOCHS="${SMOKE_EPOCHS:-1}"
  RUN_SUFFIX="smoke"
else
  EPOCHS="${EPOCHS:-20}"
  RUN_SUFFIX="main"
fi

DATASET_NAME="${DATASET_NAME:-underwater-segmentation}"
DATA_DIR="${DATA_DIR:-/kaggle/input/${DATASET_NAME}/underwater_data}"

TRAIN_IMAGES_DIR="${TRAIN_IMAGES_DIR:-${DATA_DIR}/train/images}"
TRAIN_MASKS_DIR="${TRAIN_MASKS_DIR:-${DATA_DIR}/train/masks}"

WORKING_DIR="${WORKING_DIR:-/kaggle/working}"
CHECKPOINT_DIR="${WORKING_DIR}/checkpoints"
EXPERIMENTS_DIR="${WORKING_DIR}/experiments"
ASSETS_DIR="${WORKING_DIR}/assets"
RESULTS_CSV="${EXPERIMENTS_DIR}/results.csv"

BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DEVICE="${DEVICE:-$(python - <<'PY'
try:
    import torch
    print("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    print("cpu")
PY
)}"

mkdir -p "${CHECKPOINT_DIR}" "${EXPERIMENTS_DIR}" "${ASSETS_DIR}"

if [[ ! -d "${TRAIN_IMAGES_DIR}" || ! -d "${TRAIN_MASKS_DIR}" ]]; then
  echo "Dataset directories were not found."
  echo "TRAIN_IMAGES_DIR=${TRAIN_IMAGES_DIR}"
  echo "TRAIN_MASKS_DIR=${TRAIN_MASKS_DIR}"
  echo "Set DATASET_NAME or DATA_DIR before running this script."
  exit 1
fi

run_train() {
  local experiment_name="$1"
  local model_name="$2"
  local image_size="$3"
  local weighted_loss="$4"
  local pretrained_backbone="$5"

  local extra_args=()
  if [[ "${weighted_loss}" == "1" ]]; then
    extra_args+=(--weighted-loss)
  fi
  if [[ "${pretrained_backbone}" == "1" ]]; then
    extra_args+=(--pretrained-backbone)
  fi

  echo "Running ${experiment_name}"
  python src/train.py \
    --model "${model_name}" \
    --images-dir "${TRAIN_IMAGES_DIR}" \
    --masks-dir "${TRAIN_MASKS_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}/${experiment_name}" \
    --results-csv "${RESULTS_CSV}" \
    --assets-dir "${ASSETS_DIR}/${experiment_name}" \
    --experiment-name "${experiment_name}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --image-size "${image_size}" \
    --lr "${LR}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    "${extra_args[@]}"
}

echo "DATA_DIR=${DATA_DIR}"
echo "WORKING_DIR=${WORKING_DIR}"
echo "EPOCHS=${EPOCHS}"
echo "DEVICE=${DEVICE}"
echo "RESULTS_CSV=${RESULTS_CSV}"

run_train "unet_size128_${RUN_SUFFIX}" "unet" "128" "0" "0"
run_train "unet_size256_${RUN_SUFFIX}" "unet" "256" "0" "0"
run_train "unet_weighted_ce_size256_${RUN_SUFFIX}" "unet" "256" "1" "0"
run_train "deeplabv3_resnet50_size256_${RUN_SUFFIX}" "deeplabv3_resnet50" "256" "0" "0"
run_train "deeplabv3_resnet50_pretrained_size256_${RUN_SUFFIX}" "deeplabv3_resnet50" "256" "0" "1"

echo "Done. Outputs:"
echo "- checkpoints: ${CHECKPOINT_DIR}"
echo "- results: ${RESULTS_CSV}"
echo "- assets: ${ASSETS_DIR}"
