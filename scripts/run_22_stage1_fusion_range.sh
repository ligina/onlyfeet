#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

mkdir -p logs_m14 models_m14_rgb64/stage1 reports_m14_rgb64/stage1

PY=python
SCRIPT=scripts_m14/02_train_m14_task_model.py

START_IDX=${1:-1}
END_IDX=${2:-104}

run_fusion () {
  local idx=$1
  local task=$2
  local mods=$3
  local fusion=$4

  local safe_mods=${mods//,/_}
  local data_dir="datasets_m14_rgb64_stage1/${task}"
  local run_id="stage1_${task}_${safe_mods}_${fusion}_seed42"
  local out_dir="models_m14_rgb64/stage1/${run_id}"
  local log_file="logs_m14/${run_id}.log"

  echo "======================================================================"
  echo "[JOB ${idx}] ${run_id}"
  echo "======================================================================"

  if [ -f "${out_dir}/metrics.json" ]; then
    echo "[SKIP DONE] ${run_id}"
    return 0
  fi

  if [ -d "${out_dir}" ]; then
    echo "[MOVE INCOMPLETE] ${run_id}"
    mkdir -p trash_m14_incomplete
    mv "${out_dir}" "trash_m14_incomplete/${run_id}_$(date +%Y%m%d_%H%M%S)" || true
    rm -f "${log_file}"
  fi

  echo "[RUN] ${run_id}"

  ${PY} ${SCRIPT} \
    --task "${task}" \
    --modalities "${mods}" \
    --fusion "${fusion}" \
    --train_npz "${data_dir}/dataset_train.npz" \
    --eval_npz "${data_dir}/dataset_val.npz" \
    --card "${data_dir}/dataset_card.json" \
    --out_dir "${out_dir}" \
    --epochs 100 \
    --batch 32 \
    --lr 0.0005 \
    --dropout 0.40 \
    --patience 16 \
    --label_smoothing 0.05 \
    --seed 42 \
    > "${log_file}" 2>&1

  echo "[DONE JOB] ${run_id}"
  sleep 5
}

COMBOS=(
  "imu,image"
  "imu,audio"
  "imu,tof"
  "imu,mag"
  "image,audio"
  "image,tof"
  "image,mag"
  "audio,tof"
  "audio,mag"
  "tof,mag"

  "imu,image,audio"
  "imu,image,tof"
  "imu,image,mag"
  "imu,audio,tof"
  "imu,audio,mag"
  "imu,tof,mag"
  "image,audio,tof"
  "image,audio,mag"
  "image,tof,mag"
  "audio,tof,mag"

  "imu,image,audio,tof"
  "imu,image,audio,mag"
  "imu,image,tof,mag"
  "imu,audio,tof,mag"
  "image,audio,tof,mag"

  "imu,image,audio,tof,mag"
)

idx=0

for task in activity surface; do
  for mods in "${COMBOS[@]}"; do
    for fusion in concat gated; do
      idx=$((idx + 1))

      if [ "${idx}" -lt "${START_IDX}" ]; then
        continue
      fi

      if [ "${idx}" -gt "${END_IDX}" ]; then
        echo "[STOP] requested range ${START_IDX}-${END_IDX} finished."
        ${PY} scripts_m14/04_collect_m14_results.py \
          --roots models_m14_rgb64/stage1 \
          --out_csv reports_m14_rgb64/stage1/summary_stage1_partial.csv \
          --out_json reports_m14_rgb64/stage1/summary_stage1_partial.json
        exit 0
      fi

      run_fusion "${idx}" "${task}" "${mods}" "${fusion}"

      ${PY} scripts_m14/04_collect_m14_results.py \
        --roots models_m14_rgb64/stage1 \
        --out_csv reports_m14_rgb64/stage1/summary_stage1_partial.csv \
        --out_json reports_m14_rgb64/stage1/summary_stage1_partial.json \
        > /dev/null 2>&1 || true
    done
  done
done

${PY} scripts_m14/04_collect_m14_results.py \
  --roots models_m14_rgb64/stage1 \
  --out_csv reports_m14_rgb64/stage1/summary_stage1_full.csv \
  --out_json reports_m14_rgb64/stage1/summary_stage1_full.json

echo "[DONE] requested range ${START_IDX}-${END_IDX} finished."
