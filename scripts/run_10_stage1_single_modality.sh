#!/usr/bin/env bash
set -euo pipefail
cd /workspace
mkdir -p logs_m14 models_m14_rgb64/stage1 reports_m14_rgb64/stage1

PY=python
SCRIPT=scripts_m14/02_train_m14_task_model.py

run_single () {
  local task=$1
  local mod=$2
  local data_dir="datasets_m14_rgb64_stage1/${task}"
  local run_id="stage1_${task}_${mod}_single_seed42"
  echo "[RUN] ${run_id}"
  ${PY} ${SCRIPT} \
    --task "${task}" \
    --modalities "${mod}" \
    --fusion single \
    --train_npz "${data_dir}/dataset_train.npz" \
    --eval_npz "${data_dir}/dataset_val.npz" \
    --card "${data_dir}/dataset_card.json" \
    --out_dir "models_m14_rgb64/stage1/${run_id}" \
    --epochs 80 \
    --batch 32 \
    --lr 0.0005 \
    --dropout 0.35 \
    --patience 14 \
    --label_smoothing 0.05 \
    --seed 42 \
    > "logs_m14/${run_id}.log" 2>&1
}

for mod in imu image audio tof mag; do
  run_single activity ${mod}
done

for mod in image audio imu tof mag; do
  run_single surface ${mod}
done

${PY} scripts_m14/04_collect_m14_results.py \
  --roots models_m14_rgb64/stage1 \
  --out_csv reports_m14_rgb64/stage1/summary_single_modality.csv \
  --out_json reports_m14_rgb64/stage1/summary_single_modality.json
