#!/usr/bin/env bash
set -u

cd /workspace

START=${1:-1}
END=${2:-10}

mkdir -p logs_m14

for i in $(seq "${START}" "${END}"); do
  echo "======================================================================"
  echo "[LOOP] Running fusion job ${i}"
  echo "======================================================================"

  bash scripts_m14/run_23_one_fusion_job.sh "${i}" \
    > "logs_m14/fusion_job_$(printf "%03d" ${i})_master.log" 2>&1

  status=$?

  if [ "${status}" -ne 0 ]; then
    echo "[JOB FAILED] ${i}, status=${status}"
    echo "[CONTINUE] moving to next job"
  else
    echo "[JOB DONE] ${i}"
  fi

  sleep 5
done

python scripts_m14/04_collect_m14_results.py \
  --roots models_m14_rgb64/stage1 \
  --out_csv reports_m14_rgb64/stage1/summary_stage1_partial.csv \
  --out_json reports_m14_rgb64/stage1/summary_stage1_partial.json

echo "[CURRENT COUNT]"
find models_m14_rgb64/stage1 -name metrics.json | wc -l
