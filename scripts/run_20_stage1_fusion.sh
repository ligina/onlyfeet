#!/usr/bin/env bash
set -euo pipefail
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
  echo "[RUN] ${run_id}"
  ${PY} ${SCRIPT} \
    --task "${task}" \
    --modalities "${mods}" \
    --fusion "${fusion}" \
    --train_npz "${data_dir}/dataset_train.npz" \
    --eval_npz "${data_dir}/dataset_val.npz" \
    --card "${data_dir}/dataset_card.json" \
    --out_dir "models_m14_rgb64/stage1/${run_id}" \
    --epochs 100 \
    --batch 32 \
    --lr 0.0005 \
    --dropout 0.40 \
    --patience 16 \
    --label_smoothing 0.05 \
    --seed 42 \
    > "logs_m14/${run_id}.log" 2>&1
}

# Activity fusion matrix
for fusion in concat gated; do
  run_fusion activity "imu,image" ${fusion}
  run_fusion activity "imu,audio" ${fusion}
  run_fusion activity "image,audio" ${fusion}
  run_fusion activity "imu,image,audio" ${fusion}
  run_fusion activity "imu,tof" ${fusion}
  run_fusion activity "imu,mag" ${fusion}
  run_fusion activity "imu,image,audio,tof,mag" ${fusion}
done

# Surface fusion matrix
for fusion in concat gated; do
  run_fusion surface "image,audio" ${fusion}
  run_fusion surface "image,imu" ${fusion}
  run_fusion surface "audio,imu" ${fusion}
  run_fusion surface "image,audio,imu" ${fusion}
  run_fusion surface "image,tof" ${fusion}
  run_fusion surface "audio,tof" ${fusion}
  run_fusion surface "image,audio,tof" ${fusion}
  run_fusion surface "image,audio,imu,tof,mag" ${fusion}
done

${PY} scripts_m14/04_collect_m14_results.py \
  --roots models_m14_rgb64/stage1 \
  --out_csv reports_m14_rgb64/stage1/summary_all_stage1.csv \
  --out_json reports_m14_rgb64/stage1/summary_all_stage1.json
