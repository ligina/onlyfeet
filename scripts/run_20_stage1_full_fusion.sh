#!/usr/bin/env bash
set -euo pipefail
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

cd /workspace

mkdir -p logs_m14 models_m14_rgb64/stage1 reports_m14_rgb64/stage1

PY=python
SCRIPT=scripts_m14/02_train_m14_task_model.py

run_fusion () {
  local task=$1
  local mods=$2
  local fusion=$3
  local safe_mods=${mods//,/_}
  local data_dir="datasets_m14_rgb64_stage1/${task}"
  local run_id="stage1_${task}_${safe_mods}_${fusion}_seed42"
  local out_dir="models_m14_rgb64/stage1/${run_id}"
  local log_file="logs_m14/${run_id}.log"

  if [ -f "${out_dir}/metrics.json" ]; then
    echo "[SKIP DONE] ${run_id}"
    return 0
  fi

  if [ -d "${out_dir}" ]; then
    echo "[REMOVE INCOMPLETE] ${run_id}"
    rm -rf "${out_dir}"
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

  echo "[SLEEP] releasing resources after ${run_id}"
  sleep 8
  nvidia-smi > /dev/null 2>&1 || true
}

# All modality combinations, size 2 to 5.
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

for task in activity surface; do
  for mods in "${COMBOS[@]}"; do
    run_fusion "${task}" "${mods}" "concat"
    run_fusion "${task}" "${mods}" "gated"

    ${PY} scripts_m14/04_collect_m14_results.py \
      --roots models_m14_rgb64/stage1 \
      --out_csv reports_m14_rgb64/stage1/summary_stage1_full_running.csv \
      --out_json reports_m14_rgb64/stage1/summary_stage1_full_running.json \
      > /dev/null 2>&1 || true
  done
done

${PY} scripts_m14/04_collect_m14_results.py \
  --roots models_m14_rgb64/stage1 \
  --out_csv reports_m14_rgb64/stage1/summary_stage1_full.csv \
  --out_json reports_m14_rgb64/stage1/summary_stage1_full.json

echo "[DONE] Full Stage 1 fusion experiments finished."
echo "reports_m14_rgb64/stage1/summary_stage1_full.csv"
echo "reports_m14_rgb64/stage1/summary_stage1_full.json"
