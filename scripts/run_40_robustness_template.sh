#!/usr/bin/env bash
set -euo pipefail
cd /workspace
mkdir -p reports_m14_rgb64/robustness logs_m14

# Edit paths after Stage 2 final models are trained.
ACTIVITY_MODEL="models_m14_rgb64/stage2/stage2_activity_imu_image_gated_seed42/best_model.keras"
SURFACE_MODEL="models_m14_rgb64/stage2/stage2_surface_image_audio_gated_seed42/best_model.keras"

python scripts_m14/03_eval_m14_model_robustness.py \
  --model "${ACTIVITY_MODEL}" \
  --eval_npz datasets_m14_rgb64_stage2/activity/dataset_test.npz \
  --card datasets_m14_rgb64_stage2/activity/dataset_card.json \
  --task activity \
  --out_dir reports_m14_rgb64/robustness/activity_final_seed42 \
  --conditions normal no_imu no_image no_audio no_tof no_mag \
  > logs_m14/robustness_activity_final_seed42.log 2>&1

python scripts_m14/03_eval_m14_model_robustness.py \
  --model "${SURFACE_MODEL}" \
  --eval_npz datasets_m14_rgb64_stage2/surface/dataset_test.npz \
  --card datasets_m14_rgb64_stage2/surface/dataset_card.json \
  --task surface \
  --out_dir reports_m14_rgb64/robustness/surface_final_seed42 \
  --conditions normal no_image no_audio no_imu no_tof no_mag \
  > logs_m14/robustness_surface_final_seed42.log 2>&1
