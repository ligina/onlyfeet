#!/usr/bin/env bash
set -euo pipefail
cd /workspace
mkdir -p logs_m14 models_m14_rgb64/stage2 reports_m14_rgb64/stage2

# Edit these two lines after Stage 1 results are summarized.
ACTIVITY_FINAL_MODS="imu,image"
ACTIVITY_FINAL_FUSION="gated"   # single / concat / gated. If ACTIVITY_FINAL_MODS has one modality, use single.
SURFACE_FINAL_MODS="image,audio"
SURFACE_FINAL_FUSION="gated"

PY=python
SCRIPT=scripts_m14/02_train_m14_task_model.py

run_stage2 () {
  local task=$1
  local mods=$2
  local fusion=$3
  local safe_mods=${mods//,/_}
  local data_dir="datasets_m14_rgb64_stage2/${task}"
  for seed in 42 123 2026; do
    local run_id="stage2_${task}_${safe_mods}_${fusion}_seed${seed}"
    echo "[RUN] ${run_id}"
    ${PY} ${SCRIPT} \
      --task "${task}" \
      --modalities "${mods}" \
      --fusion "${fusion}" \
      --train_npz "${data_dir}/dataset_train.npz" \
      --eval_npz "${data_dir}/dataset_test.npz" \
      --card "${data_dir}/dataset_card.json" \
      --out_dir "models_m14_rgb64/stage2/${run_id}" \
      --epochs 100 \
      --batch 32 \
      --lr 0.0005 \
      --dropout 0.40 \
      --patience 16 \
      --label_smoothing 0.05 \
      --seed ${seed} \
      > "logs_m14/${run_id}.log" 2>&1
  done
}

run_stage2 activity "${ACTIVITY_FINAL_MODS}" "${ACTIVITY_FINAL_FUSION}"
run_stage2 surface "${SURFACE_FINAL_MODS}" "${SURFACE_FINAL_FUSION}"

${PY} scripts_m14/04_collect_m14_results.py \
  --roots models_m14_rgb64/stage2 \
  --out_csv reports_m14_rgb64/stage2/summary_stage2_final.csv \
  --out_json reports_m14_rgb64/stage2/summary_stage2_final.json
