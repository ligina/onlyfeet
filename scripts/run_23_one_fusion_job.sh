#!/usr/bin/env bash
set -u

cd /workspace

export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

mkdir -p logs_m14 models_m14_rgb64/stage1 reports_m14_rgb64/stage1 locks_m14 trash_m14_incomplete

JOB_IDX=${1:-}

if [ -z "${JOB_IDX}" ]; then
  echo "Usage: bash scripts_m14/run_23_one_fusion_job.sh <job_index: 1-104>"
  exit 1
fi

exec 9>locks_m14/fusion_gpu.lock
if ! flock -n 9; then
  echo "[LOCKED] Another fusion job is already running. Exit."
  exit 2
fi

PY=python
SCRIPT=scripts_m14/02_train_m14_task_model.py

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

if [ "${JOB_IDX}" -lt 1 ] || [ "${JOB_IDX}" -gt 104 ]; then
  echo "[ERROR] JOB_IDX must be 1..104"
  exit 1
fi

zero_based=$((JOB_IDX - 1))

if [ "${JOB_IDX}" -le 52 ]; then
  task="activity"
  local_idx="${zero_based}"
else
  task="surface"
  local_idx=$((zero_based - 52))
fi

combo_idx=$((local_idx / 2))
fusion_idx=$((local_idx % 2))

mods="${COMBOS[$combo_idx]}"

if [ "${fusion_idx}" -eq 0 ]; then
  fusion="concat"
else
  fusion="gated"
fi

safe_mods=${mods//,/_}
data_dir="datasets_m14_rgb64_stage1/${task}"
run_id="stage1_${task}_${safe_mods}_${fusion}_seed42"
out_dir="models_m14_rgb64/stage1/${run_id}"
log_file="logs_m14/${run_id}.log"

echo "======================================================================"
echo "[JOB ${JOB_IDX}/104]"
echo "task       = ${task}"
echo "modalities = ${mods}"
echo "fusion     = ${fusion}"
echo "run_id     = ${run_id}"
echo "======================================================================"

if [ -f "${out_dir}/metrics.json" ]; then
  echo "[SKIP DONE] ${run_id}"
  exit 0
fi

if [ -d "${out_dir}" ]; then
  echo "[ARCHIVE INCOMPLETE] ${run_id}"
  mv "${out_dir}" "trash_m14_incomplete/${run_id}_$(date +%Y%m%d_%H%M%S)" || {
    echo "[ERROR] Cannot archive incomplete directory. Is another process using it?"
    exit 3
  }
fi

rm -f "${log_file}"

echo "[START TRAIN] ${run_id}"

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

status=$?

if [ "${status}" -ne 0 ]; then
  echo "[FAILED] ${run_id}, exit_code=${status}"
  echo "[LOG] tail -n 120 ${log_file}"
  exit "${status}"
fi

if [ -f "${out_dir}/metrics.json" ]; then
  echo "[DONE] ${run_id}"
else
  echo "[FAILED] Training exited but metrics.json not found: ${run_id}"
  exit 4
fi

${PY} scripts_m14/04_collect_m14_results.py \
  --roots models_m14_rgb64/stage1 \
  --out_csv reports_m14_rgb64/stage1/summary_stage1_partial.csv \
  --out_json reports_m14_rgb64/stage1/summary_stage1_partial.json \
  > /dev/null 2>&1 || true

echo "[COUNT]"
find models_m14_rgb64/stage1 -name metrics.json | wc -l
