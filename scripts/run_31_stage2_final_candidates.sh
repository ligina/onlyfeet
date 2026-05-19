#!/usr/bin/env bash
set -u

cd /workspace

export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1

mkdir -p models_m14_rgb64/stage2 reports_m14_rgb64/stage2 logs_m14_stage2 trash_m14_stage2_incomplete

PY=python
SCRIPT=scripts_m14/02_train_m14_task_model.py

run_stage2 () {
  local task=$1
  local mods=$2
  local fusion=$3
  local dropout=$4
  local seed=$5

  local safe_mods=${mods//,/_}
  local data_dir="datasets_m14_rgb64_stage2/${task}"
  local run_id="stage2_${task}_${safe_mods}_${fusion}_seed${seed}"
  local out_dir="models_m14_rgb64/stage2/${run_id}"
  local log_file="logs_m14_stage2/${run_id}.log"

  echo "======================================================================"
  echo "[RUN] ${run_id}"
  echo "======================================================================"

  if [ -f "${out_dir}/metrics.json" ]; then
    echo "[SKIP DONE] ${run_id}"
    return 0
  fi

  if [ -d "${out_dir}" ]; then
    echo "[ARCHIVE INCOMPLETE] ${run_id}"
    mv "${out_dir}" "trash_m14_stage2_incomplete/${run_id}_$(date +%Y%m%d_%H%M%S)" || true
    rm -f "${log_file}"
  fi

  ${PY} ${SCRIPT} \
    --task "${task}" \
    --modalities "${mods}" \
    --fusion "${fusion}" \
    --train_npz "${data_dir}/dataset_train.npz" \
    --eval_npz "${data_dir}/dataset_test.npz" \
    --card "${data_dir}/dataset_card.json" \
    --out_dir "${out_dir}" \
    --epochs 100 \
    --batch 32 \
    --lr 0.0005 \
    --dropout "${dropout}" \
    --patience 16 \
    --label_smoothing 0.05 \
    --seed "${seed}" \
    > "${log_file}" 2>&1

  status=$?

  if [ "${status}" -ne 0 ]; then
    echo "[FAILED] ${run_id}, status=${status}"
    echo "[LOG] ${log_file}"
    return "${status}"
  fi

  if [ -f "${out_dir}/metrics.json" ]; then
    echo "[DONE] ${run_id}"
  else
    echo "[FAILED] ${run_id}: metrics.json not found"
    return 4
  fi
}

SEED=42

# Activity Stage 2 candidates
run_stage2 activity "imu" "single" 0.35 ${SEED}
run_stage2 activity "image" "single" 0.35 ${SEED}
run_stage2 activity "imu,image" "gated" 0.40 ${SEED}
run_stage2 activity "imu,image,audio" "concat" 0.40 ${SEED}
run_stage2 activity "imu,image,audio,tof" "concat" 0.40 ${SEED}

# Surface Stage 2 candidates
run_stage2 surface "image" "single" 0.35 ${SEED}
run_stage2 surface "audio" "single" 0.35 ${SEED}
run_stage2 surface "image,tof" "concat" 0.40 ${SEED}
run_stage2 surface "image,tof" "gated" 0.40 ${SEED}
run_stage2 surface "imu,image,tof" "concat" 0.40 ${SEED}
run_stage2 surface "image,audio" "concat" 0.40 ${SEED}
run_stage2 surface "image,audio,tof" "concat" 0.40 ${SEED}

${PY} scripts_m14/04_collect_m14_results.py \
  --roots models_m14_rgb64/stage2 \
  --out_csv reports_m14_rgb64/stage2/summary_stage2_final_candidates.csv \
  --out_json reports_m14_rgb64/stage2/summary_stage2_final_candidates.json

echo "[DONE] Stage 2 final candidates complete."
