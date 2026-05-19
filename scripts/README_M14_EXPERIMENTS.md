# OnlyFeet M14 Experiment Scripts

Copy these files into:

```bash
/home/jovyan/work/bs/final/scripts_m14/
```

Then run from the project root:

```bash
cd /home/jovyan/work/bs/final
```

## Files

- `02_train_m14_task_model.py`  
  Universal task-specialist trainer. Handles single modality, concat fusion, and gated fusion.

- `03_eval_m14_model_robustness.py`  
  Evaluates a trained model under normal and missing-modality conditions.

- `04_collect_m14_results.py`  
  Collects all `metrics.json` files into summary CSV/JSON.

- `run_10_stage1_single_modality.sh`  
  Runs all Stage 1 single-modality baselines.

- `run_20_stage1_fusion.sh`  
  Runs broad Stage 1 concat/gated multimodal fusion experiments.

- `run_30_stage2_template.sh`  
  Template for final Stage 2 P4 held-out test. Edit final modality settings after Stage 1.

- `run_40_robustness_template.sh`  
  Template for missing-modality robustness evaluation. Edit model paths after Stage 2.

## Recommended order

```bash
chmod +x scripts_m14/*.sh scripts_m14/*.py
bash scripts_m14/run_10_stage1_single_modality.sh
bash scripts_m14/run_20_stage1_fusion.sh
python scripts_m14/04_collect_m14_results.py --roots models_m14_rgb64/stage1 --out_csv reports_m14_rgb64/stage1/summary_all_stage1.csv --out_json reports_m14_rgb64/stage1/summary_all_stage1.json
```

After inspecting Stage 1 summaries, edit `run_30_stage2_template.sh`, then run:

```bash
bash scripts_m14/run_30_stage2_template.sh
```

After Stage 2 final models are available, edit `run_40_robustness_template.sh`, then run:

```bash
bash scripts_m14/run_40_robustness_template.sh
```

## Notes

- Stage 1 uses P1and2 for training and P3 for validation.
- Stage 2 uses P1and2 + P3 for training and P4 as held-out test.
- Surface recognition uses walking-only samples.
- Primary selection metric: macro-F1.
- Do not use P4 for Stage 1 model selection.
