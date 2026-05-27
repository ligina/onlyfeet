#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

mkdir -p models/clean_p4_final logs_clean_p4_final

PY="${PY:-python}"
SCRIPT="scripts/02_train_m14_task_model_clean_p4.py"

run_clean_p4 () {
  local task=$1
  local mods=$2
  local fusion=$3
  local dropout=$4
  local seed=$5

  local safe_mods=${mods//,/_}
  local data_dir="datasets_m14_rgb64_stage2/${task}"
  local run_id="stage2_${task}_${safe_mods}_${fusion}_seed${seed}_cleanp4"
  local out_dir="models/clean_p4_final/${run_id}"
  local log_file="logs_clean_p4_final/${run_id}.log"

  echo "======================================================================"
  echo "[RUN CLEAN P4] ${run_id}"
  echo "======================================================================"

  if [ -e "${out_dir}" ]; then
    echo "[REFUSE OVERWRITE] ${out_dir} already exists."
    echo "Move or rename it manually before rerunning."
    return 3
  fi

  "${PY}" "${SCRIPT}" \
    --task "${task}" \
    --modalities "${mods}" \
    --fusion "${fusion}" \
    --train_npz "${data_dir}/dataset_train.npz" \
    --test_npz "${data_dir}/dataset_test.npz" \
    --card "${data_dir}/dataset_card.json" \
    --out_dir "${out_dir}" \
    --epochs 100 \
    --batch 32 \
    --lr 0.0005 \
    --dropout "${dropout}" \
    --patience 16 \
    --label_smoothing 0.05 \
    --seed "${seed}" \
    --validation_split_ratio 0.15 \
    > "${log_file}" 2>&1

  if [ -f "${out_dir}/metrics.json" ]; then
    echo "[DONE CLEAN P4] ${run_id}"
  else
    echo "[FAILED] ${run_id}: metrics.json not found"
    echo "[LOG] ${log_file}"
    return 4
  fi
}

SEED=42

run_clean_p4 activity "imu" "single" 0.35 "${SEED}"
run_clean_p4 surface "image,audio" "concat" 0.40 "${SEED}"

echo "[DONE] Clean-P4 Stage 2 final-model reruns complete."
