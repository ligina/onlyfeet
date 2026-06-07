# OnlyFeet Final Experiment Release

[English](README.md) | [中文](README.zh-CN.md)

This repository contains the final experiment release for the **OnlyFeet bachelor thesis project**. It includes training scripts, evaluation scripts, result reports, selected trained models, logs, dataset and split documentation, archived original artifacts, and environment information.

Internally, the final held-out P4 evaluation protocol is named Clean-P4. The current thesis main final evidence is the **Clean-P4 final evaluation**. Clean-P4 corrects the earlier Stage 2 checkpoint-selection risk by deriving training-time validation from the Stage 2 training data only and reserving P4 for final evaluation only.

The project investigates **multimodal human activity recognition (HAR)** and **walking-only surface recognition** using the existing OnlyFeet foot-mounted acquisition prototype. The available sensing modalities are:

* IMU
* RGB image
* audio
* Time-of-Flight (ToF)
* magnetometer

The main goal is to evaluate task-specific modality usefulness and fusion behavior under a controlled participant split. The release does not claim deployment readiness, measured latency, energy efficiency, battery runtime, or broad real-world robustness.

---

## Dataset Availability

The public OnlyFeet dataset archive is hosted on Hugging Face:

[OnlyFeet Dataset on Hugging Face](https://huggingface.co/datasets/ligina592/onlyfeet/tree/main)

Code and experiment scripts are hosted in this GitHub repository. The large dataset archive is hosted separately on Hugging Face. This separation keeps the Git repository lightweight and avoids storing large binary data in Git.

The GitHub repository does not store the large raw/final dataset archive directly. Dataset metadata such as `dataset_card.json` and `build_stats.json` may be included in the repository for traceability, while the full uploaded dataset archive is available from the Hugging Face dataset repository.

Current Hugging Face dataset contents include `data_makeup.zip` (approximately 1.13 GB). The license shown on Hugging Face is MIT. Users should download the dataset archive from Hugging Face and place or extract it according to the expected local project/data layout before running the dataset-preparation or training scripts.

```bash
# Download the dataset archive from Hugging Face, then extract it into
# the expected local data directory before running scripts.
huggingface-cli download ligina592/onlyfeet data_makeup.zip --repo-type dataset --local-dir data_external/onlyfeet
```

---

## 1. Repository Structure

```text
onlyfeet/
├── archive_manifest.txt
├── analysis_outputs/
├── datasets_m14_rgb64_stage1/
├── datasets_m14_rgb64_stage2/
├── data_docs/
├── environment/
├── logs/
│   ├── logs_clean_p4_final/
│   ├── stage2/
│   ├── robustness_trainnorm/
│   └── fusion_strategy/
├── models/
│   ├── clean_p4_final/
│   └── final_stage2/
├── requirement/
├── reports/
├── scripts/
├── README.md
└── README.zh-CN.md
```

### Main directories

```text
scripts/
```

Contains dataset preparation, model training, evaluation, result collection, robustness testing, folder-level evaluation, sanity-check scripts, and the Clean-P4 rerun scripts.

```text
analysis_outputs/
```

Contains supplementary analysis outputs that are useful for traceability, including Clean-P4 supplementary-check summaries and sample-difference diagnostics. These files are supporting diagnostics and do not replace the main Clean-P4 final evidence.

```text
datasets_m14_rgb64_stage1/
datasets_m14_rgb64_stage2/
```

Contain dataset metadata cards and build statistics for the generated M14 RGB64 Stage 1 and Stage 2 datasets. The large `.npz` array files required for full training reruns are not included in this release.

```text
reports/
```

Contains Stage 1 summaries, original Stage 2 summaries, final-evaluation tables, robustness reports, folder-level evaluation, fusion-strategy comparison, non-overlap-window reports, and split sanity checks. Reports generated under the original Stage 2 protocol should be interpreted with the deprecated/original caveats below unless rerun under Clean-P4.

```text
models/
```

Contains both the Clean-P4 final models and the older original Stage 2 models:

* Clean-P4 main final models: `models/clean_p4_final/`
* Deprecated/original Stage 2 traceability models: `models/final_stage2/`

```text
logs/
```

Contains key training and evaluation logs, including Clean-P4 final rerun logs in `logs/logs_clean_p4_final/`.

```text
data_docs/
```

Contains dataset and split documentation, including participant definitions and split rules.

```text
requirement/
```

Contains thesis requirement or submission-reference material, including `requirement/Ziang Liu.pdf`.

```text
environment/
```

Contains environment information, including Python version, installed packages, GPU-environment export files, and storage information.

---

## 2. Clean-P4 Main Evidence Boundary

Clean-P4 is the thesis main final-result evidence. Under Clean-P4:

* Stage 2 training data are used to fit model parameters.
* Internal validation is derived from the Stage 2 training NPZ only.
* `ModelCheckpoint`, `ReduceLROnPlateau`, and `EarlyStopping` monitor internal validation metrics, not P4 metrics.
* P4 is loaded only after training and checkpoint selection are complete.
* P4 is used for final evaluation only.

The original `final_stage2` outputs are retained in the release for traceability, but they are deprecated/original because P4 was used as Keras `validation_data` during training. That allowed P4-derived validation metrics to influence checkpoint selection, early stopping, or learning-rate scheduling.

---

## 3. Thesis Main Final Results

The thesis main final evidence consists of two task-specific Clean-P4 final models.

| Task | Clean-P4 final model | Modalities | Fusion | Rounded P4 result |
|---|---|---|---|---|
| Activity recognition | `stage2_activity_imu_single_seed42_cleanp4` | IMU | single | about 99.96% accuracy and about 99.96% macro-F1 |
| Walking-only surface recognition | `stage2_surface_image_audio_concat_seed42_cleanp4` | image, audio | concat | about 95.06% accuracy and about 94.59% macro-F1 |

These results should be interpreted as controlled subject-held-out P4 performance on the M14 dataset. Activity performance is very high, but it should not be described as perfect. Walking-only surface performance is strong, but it should not be described as near-perfect or generalized to non-walking activities.

### Clean-P4 model locations

```text
models/clean_p4_final/stage2_activity_imu_single_seed42_cleanp4/
models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4/
```

Each Clean-P4 model directory contains the trained model files, training configuration, training history, training log, final evaluation metrics, predictions, and evaluation reports.

Important Clean-P4 support files:

```text
models/clean_p4_final/
logs/logs_clean_p4_final/
scripts/02_train_m14_task_model_clean_p4.py
scripts/run_32_stage2_clean_p4_final_models.sh
```

---

## Clean-P4 Supplementary Diagnostics

The following supplementary diagnostics were recomputed under the Clean-P4 evidence chain:

* Non-overlapping-window evaluation:
  * Activity: 2,735 -> 1,417 windows, 99.93% macro-F1
  * Surface: 951 -> 493 windows, 95.53% macro-F1

* Surface missing-modality ablation:
  * Normal image+audio: 94.59% macro-F1
  * No image: 29.25% macro-F1
  * No audio: 19.17% macro-F1

* Model complexity:
  * Activity IMU specialist: 99,139 parameters
  * Surface image+audio specialist: 280,613 parameters
  * Composite wrapper: 379,752 parameters

These diagnostics are supplementary and do not replace the Clean-P4 main final models.

Report locations:

```text
Non-overlap report: reports/clean_p4_final/non_overlap/
Surface robustness report: reports/clean_p4_final/robustness_surface/
Model complexity report: reports/clean_p4_final/model_complexity/
Composite wrapper report: reports/clean_p4_final/unified_composite_cleanp4/
```

Interpretation boundary: Non-overlap supports that the high result is not solely caused by overlapping-window redundancy. Surface ablation shows image and audio are both important, but the current model is not robust to complete single-modality failure. The composite wrapper is an engineering packaging artifact and is not a jointly trained multitask model.

---

## 4. Project Goal

The goal of this project is to evaluate multimodal recognition using the OnlyFeet foot-mounted acquisition prototype.

The system targets two main tasks:

1. **Regular activity recognition**

   * walk
   * standing
   * sitting

2. **Walking-only surface recognition**

   * asphalt
   * PVC
   * sand
   * gravel
   * grass

Surface recognition is based on **walking-only samples**, because foot-ground interaction cues are primarily available during walking. Surface results in this release should not be interpreted as general surface recognition across standing, sitting, stair movement, or other untested activities.

---

## 5. Dataset Summary

The cleaned M14 dataset contains:

```text
Total folder-level recordings: 1,166
P1/P2: 691 folders
P3: 369 folders
P4: 106 folders
```

### Activity classes

```text
walk
standing
sitting
```

### Surface classes

```text
asphalt
PVC
sand
gravel
grass
```

### Modalities

Each recording folder may contain synchronized multimodal data from:

```text
pkt_*.json   sensor packets: IMU, ToF, magnetometer
rec_*.wav    audio recordings
img_*.jpg    camera image stream
mid_*.jpg    second image stream
```

The generated window datasets contain fields such as:

```text
imu_win
tof_win
mag_win
audio_win
img_win
y_act
y_env
folder
participant
start_ms
```

### Feature Representation Notes

The raw IMU window stored in the prepared dataset is `imu_win` with shape `[50, 6]`, corresponding to accelerometer and gyroscope axes. During model loading, the training and evaluation scripts append two derived magnitude channels: accelerometer norm and gyroscope norm. Therefore the model-side IMU input has shape `[50, 8]`. This is expected and does not indicate a dataset/model mismatch.

This expansion is performed by `add_imu_magnitudes()` in the training and evaluation scripts.

---

## 6. Participant Definition

The participant labels are defined as follows.

### P1/P2

`P1/P2` is a merged participant group from early recordings collected by Ziang and another student.

During early data collection, there were some recording issues, including battery depletion and partially missing recordings. Because of this, some recordings could no longer be reliably separated into P1 or P2 during later dataset cleanup. Therefore, these recordings are merged as one group: `P1/P2`.

### P3

`P3` is a separate participant recorded on a different date.

### P4

`P4` is another separate participant recorded on a different date. P4 is used as the final held-out participant test set in Stage 2 and Clean-P4. Under Clean-P4, P4 is final-evaluation-only.

---

## 7. Experimental Design

The experiments are organized into Stage 1 model search, original Stage 2 traceability, and Clean-P4 final evaluation.

### 7.1 Stage 1: Model Search

Stage 1 is used for model selection, modality comparison, and fusion-strategy comparison.

```text
Train: P1/P2
Validation: P3
```

In Stage 1, 114 models were trained:

```text
10 single-modality baselines
104 multimodal fusion models
```

The evaluated modalities include:

```text
IMU
image
audio
ToF
magnetometer
```

The evaluated fusion methods include:

```text
concat fusion
gated fusion
```

Stage 1 is the systematic model-search stage.

### 7.2 Original Stage 2: Deprecated/Original Traceability

Original Stage 2 used:

```text
Train: P1/P2 + P3
Test: P4
```

It evaluated 12 pre-selected candidate models on P4. However, the original Stage 2 training script also used P4 `dataset_test.npz` as Keras `validation_data`. As a result, P4-derived `val_loss` could influence `ModelCheckpoint`, `ReduceLROnPlateau`, and `EarlyStopping`.

For this reason, original Stage 2 outputs are retained only as deprecated/original traceability artifacts. They should not be used as the thesis main final evidence and should not be described as clean held-out evaluation.

### 7.3 Clean-P4 Final Evaluation

Clean-P4 keeps the two selected final configurations but corrects the validation role:

```text
Training source: Stage 2 dataset_train.npz
Internal validation: folder-level split from Stage 2 dataset_train.npz
Final test: Stage 2 dataset_test.npz, corresponding to P4
```

P4 is not used for training, validation, checkpoint selection, early stopping, or learning-rate scheduling under Clean-P4.

### Stage 2 dataset sizes

Activity recognition:

```text
Train: 29,788 windows, 1,058 folders
Test: 2,735 windows, 105 folders
```

Walking-only surface recognition:

```text
Train: 10,299 windows, 371 folders
Test: 951 windows, 36 folders
```

---

## Training and Evaluation Architecture

Dataset preparation converts raw multimodal recording folders into window-level NPZ datasets. Each 2-second window contains synchronized modality representations. Activity labels are used for walk, standing, and sitting. Surface labels are evaluated only for walking windows.

The model-search stage uses P1/P2 for training and P3 for validation. It compares single-modality baselines and multimodal fusion candidates. This stage supports model and modality selection, but it is not the final thesis evidence.

In the final training stage, P1/P2 and P3 are combined as training data and the selected final configurations are retrained. Under Clean-P4, internal validation is derived only from the training data using a folder-level internal split.

For final held-out evaluation, P4 is loaded only after training and checkpoint selection are complete. P4 is used only for final evaluation. The two final task-specific models are the activity specialist, which is IMU-only, and the walking-only surface specialist, which uses image+audio concat fusion.

Supplementary diagnostics are reported separately from the main final models. Non-overlap evaluation checks whether high performance is mainly caused by overlapping windows. Surface modality ablation checks how much the image/audio surface model depends on each modality. Model complexity reports parameter counts and file sizes. The composite wrapper is an engineering packaging artifact and is not a jointly trained multitask model.

---

## 8. Clean-P4 Final Selected Models

### 8.1 Clean-P4 Activity Model

The Clean-P4 activity model is:

```text
IMU-only
single modality
seed 42
```

Clean-P4 P4 window-level result:

```text
Accuracy: about 99.96%
Macro-F1: about 99.96%
```

The exact Clean-P4 metrics are stored in:

```text
models/clean_p4_final/stage2_activity_imu_single_seed42_cleanp4/eval_metrics.json
```

Complexity recorded for this model:

```text
Parameters: 99,139
Model size: about 1.22 MB
```

These values support discussion of model compactness, but they do not measure latency, energy consumption, battery runtime, or deployment readiness.

### 8.2 Clean-P4 Walking-Only Surface Model

The Clean-P4 walking-only surface model is:

```text
Image + Audio concat fusion
seed 42
```

Clean-P4 P4 window-level result:

```text
Accuracy: about 95.06%
Macro-F1: about 94.59%
```

The exact Clean-P4 metrics are stored in:

```text
models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4/eval_metrics.json
```

Complexity recorded for this model:

```text
Parameters: 280,613
Model size: about 3.38 MB
```

This result is limited to walking-only surface recognition over the registered M14 surface classes.

---

## 9. Clean-P4 Training and Reproduction Files

Clean-P4 training script:

```text
scripts/02_train_m14_task_model_clean_p4.py
```

Clean-P4 runner:

```text
scripts/run_32_stage2_clean_p4_final_models.sh
```

Clean-P4 model outputs:

```text
models/clean_p4_final/
```

Clean-P4 logs:

```text
logs/logs_clean_p4_final/
```

The runner executes exactly the two thesis final configurations:

```text
activity, modalities=imu, fusion=single, dropout=0.35, seed=42
surface, modalities=image,audio, fusion=concat, dropout=0.40, seed=42
```

A typical Clean-P4 rerun command is:

```bash
bash scripts/run_32_stage2_clean_p4_final_models.sh
```

A full rerun requires the prepared Stage 2 NPZ datasets, which are not included in this release.

---

## 10. Main Result Tables and Reports

The original final-evaluation tables are located in:

```text
reports/final_eval/
```

Important files:

```text
reports/final_eval/table_final_selected_models.csv
reports/final_eval/table_stage2_final_candidates_sorted.csv
reports/final_eval/table_stage1_all_114_sorted.csv
reports/final_eval/table_stage1_top20_by_task.csv
reports/final_eval/table_stage1_best_by_task.csv
reports/final_eval/table_stage1_fusion_type_summary.csv
reports/final_eval/table_robustness_trainnorm.csv
reports/final_eval/table_early_mid_late_extra.csv
reports/final_eval/table_folder_level_summary.csv
```

These tables were generated before the Clean-P4 correction unless explicitly regenerated under Clean-P4. Use the Clean-P4 model directories and logs as the main final-result evidence for the thesis. Treat original `final_stage2`, fusion, folder-level, robustness, and non-overlap outputs as deprecated/original diagnostics. Clean-P4 supplementary non-overlap, surface robustness, and model-complexity diagnostics are available under `reports/clean_p4_final/`.

---

## 11. Stage 1 Summary

Stage 1 includes the full model-selection experiment.

Location:

```text
reports/stage1/
```

Important files:

```text
summary_stage1_all_114.csv
summary_stage1_all_114.json
summary_stage1_fusion_full.csv
summary_stage1_fusion_full.json
summary_single_modality.csv
summary_single_modality.json
```

Stage 1 was used to compare:

* single-modality baselines
* multimodal concat fusion
* multimodal gated fusion
* different modality combinations
* task-specific modality usefulness

Stage 1 is model-search evidence, not final held-out P4 evidence.

---

## 12. Original Stage 2 Summary: Deprecated/Original

Original Stage 2 candidate reports are located in:

```text
reports/stage2/
```

Important files:

```text
summary_stage2_final_candidates.csv
summary_stage2_final_candidates.json
```

Original Stage 2 evaluated 12 candidate models selected after Stage 1. These reports are deprecated/original for final-result purposes because P4 was used as Keras validation data during training-time callback decisions.

Do not present the original 100.00% activity result or original 99.26% walking-only surface result as the thesis main final results.

---

## 13. Deprecated/Original Robustness Evaluation

Missing-modality robustness was evaluated using train-normalized evaluation under the original Stage 2 evidence chain.

Location:

```text
reports/robustness_trainnorm/
```

Important summary file:

```text
reports/robustness_trainnorm/summary_robustness_trainnorm_all.csv
```

These robustness outputs are deprecated/original unless rerun under Clean-P4. They may be useful as traceability diagnostics, but they should not be presented as Clean-P4 robustness evidence.

Clean-P4 surface missing-modality robustness diagnostics are now available as supplementary evidence under:

```text
reports/clean_p4_final/robustness_surface/
```

---

## 14. Deprecated/Original Fusion Strategy Comparison

Early fusion, mid-level fusion, and late fusion were compared under the original diagnostic evidence chain.

Location:

```text
reports/fusion_strategy_extra/
```

Important files:

```text
summary_stage2_early_late.csv
summary_stage2_early_late.json
```

These fusion reports are deprecated/original unless rerun under Clean-P4. They can document the development path, but they should not be used to claim Clean-P4 fusion superiority.

---

## 15. Deprecated/Original Folder-Level and Non-Overlap Diagnostics

Folder-level majority-vote reports are located in:

```text
reports/folder_level/
```

Non-overlapping-window reports are located in:

```text
reports/nonoverlap_windows/
```

The folder-level and original non-overlapping-window reports in these legacy locations are retained as deprecated/original traceability evidence. They do not replace the Clean-P4 main final models.

Clean-P4 folder-level majority-vote evaluation was not recomputed and is therefore not included in the supplementary diagnostics.

Clean-P4 non-overlap diagnostics are now available as supplementary evidence under:

```text
reports/clean_p4_final/non_overlap/
```

Clean-P4 model-complexity diagnostics are now available as supplementary evidence under:

```text
reports/clean_p4_final/model_complexity/
```

---

## 16. Split Sanity Checks

Sanity checks were performed to verify split integrity and folder overlap.

Location:

```text
reports/sanity_checks/
```

Important files:

```text
dataset_split_summary.csv
folder_overlap_summary.csv
label_csv_summary.json
participant_x_split_stage2.csv
```

Key findings:

```text
Stage 2 activity train/test folder overlap: 0
Stage 2 walking-only surface train/test folder overlap: 0
```

The activity and walking-only surface datasets overlap within the same split because walking-only surface recognition is constructed as the walking subset of the activity dataset. This is expected and is not train/test leakage.

These folder-overlap checks support the intended participant split. They do not by themselves establish broad generalization, Clean-P4 folder-level performance, Clean-P4 non-overlap performance, or real-world robustness.

---

## 17. Trained Models

### Clean-P4 main final models

```text
models/clean_p4_final/stage2_activity_imu_single_seed42_cleanp4/
models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4/
```

Each Clean-P4 model directory contains:

```text
best_model.h5
final_model.h5
metrics.json
train_config.json
history.json
training_log.csv
eval_metrics.json
eval_predictions.csv
eval_classification_report.txt
eval_confusion_matrix.csv
eval_confusion_matrix.png
eval_confusion_matrix_normalized.png
```

### Clean-P4 composite wrapper

```text
models/clean_p4_final/unified_composite_cleanp4/
```

The composite wrapper is an engineering packaging artifact and is not a jointly trained multitask model. It wraps the two final task-specific models and should not be reported as a separate scientific result.

### Deprecated/original Stage 2 models

```text
models/final_stage2/stage2_activity_imu_single_seed42/
models/final_stage2/stage2_surface_image_audio_concat_seed42/
```

These directories are retained for traceability. They are not the thesis main final evidence because the original Stage 2 training used P4-derived validation metrics for checkpoint selection, early stopping, or learning-rate scheduling.

---

## 18. Logs

Logs are stored in:

```text
logs/
```

Important subdirectories:

```text
logs/logs_clean_p4_final/
logs/stage2/
logs/robustness_trainnorm/
logs/fusion_strategy/
```

Use `logs/logs_clean_p4_final/` for the Clean-P4 final rerun. The other log directories document original Stage 2, robustness, and fusion diagnostics and should be interpreted with the deprecated/original caveats above.

---

## 19. Scripts

The main scripts are located in:

```text
scripts/
```

### 19.1 Dataset Preparation

```text
scripts/01_prepare_task_datasets_m14_rgb64.py
```

Builds Stage 1 and Stage 2 RGB64 task datasets from labeled raw folders.

### 19.2 Original Main Training

```text
scripts/02_train_m14_task_model.py
```

Original training script for single-modality, concat fusion, and gated fusion models. Original Stage 2 outputs from this path are deprecated/original for final-result purposes where P4 was used as validation data.

### 19.3 Clean-P4 Main Training

```text
scripts/02_train_m14_task_model_clean_p4.py
scripts/run_32_stage2_clean_p4_final_models.sh
```

These are the Clean-P4 scripts for the thesis main final evidence.

### 19.4 Early/Late Fusion Training

```text
scripts/02b_train_m14_early_late_fusion.py
```

Extra script for early-flat and late-average fusion experiments. Treat related outputs as deprecated/original unless rerun under Clean-P4.

### 19.5 Robustness Evaluation

```text
scripts/03_eval_m14_model_robustness.py
scripts/03b_eval_m14_model_robustness_trainnorm.py
```

The `03b` version was used for the reported original robustness tables. These outputs are deprecated/original unless rerun under Clean-P4.

### 19.6 Result Collection

```text
scripts/04_collect_m14_results.py
```

Collects model metrics into CSV/JSON summary tables.

### 19.7 Folder-Level Evaluation

```text
scripts/06_eval_folder_level_majority.py
```

Performs folder-level majority-vote evaluation from window-level predictions. Existing folder-level outputs are deprecated/original unless rerun under Clean-P4.

### 19.8 Final Table Generation

```text
scripts/07_make_final_evaluation_tables.py
```

Generates thesis-ready evaluation tables for the original report set. Use Clean-P4 outputs as the thesis main final evidence.

### 19.9 Sanity Checks

```text
scripts/08_sanity_check_m14_final.py
```

Checks dataset split summaries and folder overlap.

### 19.10 Supplementary Clean-P4 Diagnostic Scripts

```text
scripts/run_cleanp4_supplementary_checks.py
scripts/94_eval_clean_p4_non_overlap.py
scripts/95_eval_clean_p4_surface_robustness.py
scripts/96_collect_clean_p4_model_complexity.py
scripts/create_clean_p4_composite_model.py
```

`scripts/run_cleanp4_supplementary_checks.py` is a convenience runner for the Clean-P4 supplementary diagnostics and related consistency checks.

`scripts/94_eval_clean_p4_non_overlap.py` runs the Clean-P4 non-overlapping-window diagnostic and outputs to `reports/clean_p4_final/non_overlap/`.

`scripts/95_eval_clean_p4_surface_robustness.py` runs the Clean-P4 surface missing-modality ablation and outputs to `reports/clean_p4_final/robustness_surface/`.

`scripts/96_collect_clean_p4_model_complexity.py` collects the Clean-P4 model complexity summary and outputs to `reports/clean_p4_final/model_complexity/`.

`scripts/create_clean_p4_composite_model.py` creates the composite wrapper and outputs to `models/clean_p4_final/unified_composite_cleanp4/` and `reports/clean_p4_final/unified_composite_cleanp4/`.

### 19.11 Batch Runner Scripts

```text
scripts/run_10_stage1_single_modality.sh
scripts/run_20_stage1_fusion.sh
scripts/run_20_stage1_full_fusion.sh
scripts/run_22_stage1_fusion_range.sh
scripts/run_23_one_fusion_job.sh
scripts/run_24_fusion_jobs_loop.sh
scripts/run_30_stage2_template.sh
scripts/run_31_stage2_final_candidates.sh
scripts/run_32_stage2_clean_p4_final_models.sh
scripts/run_40_robustness_template.sh
```

### 19.12 Data Inspection Scripts

```text
scripts/check_m14_no_window_folders.py
scripts/inspect_p4_bad_walk_folders.py
scripts/09_eval_nonoverlap_windows.py
scripts/scan_p12_p3_imu_missing.py
scripts/scan_p4_all_activity_imu.py
scripts/scan_p4_all_imu_packets.py
```

These scripts were used for data quality inspection, legacy non-overlap diagnostics, and debugging.

---

## 20. Environment

Environment information is stored in:

```text
environment/
```

Included files:

```text
python_version.txt
pip_freeze.txt
nvidia_smi.txt
storage_info.txt
```

The main experiments were run on a cloud machine with an RTX 5090 GPU.

---

## 21. Reproducing the Experiments

The raw dataset is not included in this release due to size and privacy/storage constraints.

However, the included scripts and reports allow review of:

* split logic
* training setup
* Clean-P4 final evaluation setup
* original Stage 1 and Stage 2 summaries
* logs
* trained final Clean-P4 models
* deprecated/original diagnostics
* sanity checks

A typical full reproduction would require the original raw dataset or prepared `.npz` datasets.

### 21.1 Stage 1 Single-Modality Baselines

```bash
bash scripts/run_10_stage1_single_modality.sh
```

### 21.2 Stage 1 Full Fusion Search

```bash
bash scripts/run_20_stage1_full_fusion.sh
```

For more stable execution on limited GPU memory, jobs can be executed sequentially:

```bash
bash scripts/run_24_fusion_jobs_loop.sh 1 104
```

### 21.3 Original Stage 2 Final Candidate Evaluation

```bash
bash scripts/run_31_stage2_final_candidates.sh
```

This reproduces original Stage 2 candidate evaluation, which is deprecated/original for final-result purposes because P4 was used as validation data during training-time callback decisions.

### 21.4 Clean-P4 Final Models

```bash
bash scripts/run_32_stage2_clean_p4_final_models.sh
```

This is the runner for the thesis main final Clean-P4 models.

### 21.5 Deprecated/Original Robustness Evaluation

```bash
python scripts/03b_eval_m14_model_robustness_trainnorm.py
```

The exact commands are documented in logs and runner scripts. Treat these robustness outputs as deprecated/original unless rerun under Clean-P4.

### 21.6 Original Final Tables

```bash
python scripts/07_make_final_evaluation_tables.py
```

### 21.7 Sanity Checks

```bash
python scripts/08_sanity_check_m14_final.py
```

---

## 22. Important Caveats and Limitations

The Clean-P4 final P4 results are high:

```text
Activity: about 99.96% accuracy and about 99.96% macro-F1
Walking-only surface: about 95.06% accuracy and about 94.59% macro-F1
```

These are the thesis main final results. Do not replace them with the older original Stage 2 100.00% activity or 99.26% walking-only surface values.

Sanity checks found no Stage 2 train/test folder overlap. However, the data were collected in a relatively controlled setting.

Potential limitations:

1. The number of participants is limited.
2. P1/P2 is a merged group rather than two fully separable participants.
3. Window-level samples from the same recording are temporally correlated.
4. Walking-only surface recognition is limited to walking samples.
5. P4 data may be more standardized than fully unconstrained real-world data.
6. Participants may have performed actions more carefully and consistently during data collection.
7. Location-specific visual or acoustic cues cannot be ruled out for walking-only surface recognition.
8. Clean-P4 folder-level evaluation was not recomputed and is therefore not included in the supplementary diagnostics. Window-level Clean-P4 non-overlap, surface robustness, and model-complexity diagnostics are available under `reports/clean_p4_final/`.

For these reasons, the Clean-P4 results should be interpreted as controlled subject-held-out performance rather than full real-world generalization.

---

## 23. Recommended Interpretation

The current Clean-P4 results support the following cautious conclusions:

1. **Activity recognition is strongly supported by IMU in this controlled M14 task.**
   The Clean-P4 final activity model is an IMU-only model and performs very strongly on held-out P4 activity windows.

2. **Walking-only surface recognition benefits from image and audio in the selected final model.**
   The Clean-P4 final walking-only surface model uses image+audio concat fusion.

3. **Clean-P4 is the main final evidence boundary.**
   P4 is final-evaluation-only, and training-time validation is derived from Stage 2 training data.

4. **Original Stage 2 and original diagnostic outputs remain useful only as traceability unless rerun.**
   Original final_stage2, fusion, and folder-level diagnostics remain deprecated/original unless rerun under Clean-P4. Clean-P4 supplementary non-overlap, surface robustness, and model-complexity diagnostics are available under `reports/clean_p4_final/`.

5. **The final results should be reported with caution.**
   The evaluation is controlled and participant-held-out, but the dataset remains limited in participant diversity and recording context.

---

## 24. Key Files for Review

For Clean-P4 thesis main final evidence, start with:

```text
models/clean_p4_final/stage2_activity_imu_single_seed42_cleanp4/train_config.json
models/clean_p4_final/stage2_activity_imu_single_seed42_cleanp4/eval_metrics.json
models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4/train_config.json
models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4/eval_metrics.json
logs/logs_clean_p4_final/
scripts/02_train_m14_task_model_clean_p4.py
scripts/run_32_stage2_clean_p4_final_models.sh
```

For original traceability reports, review:

```text
reports/final_eval/table_final_selected_models.csv
reports/final_eval/table_stage2_final_candidates_sorted.csv
reports/final_eval/table_robustness_trainnorm.csv
reports/final_eval/table_early_mid_late_extra.csv
reports/final_eval/table_folder_level_summary.csv
reports/nonoverlap_windows/summary_nonoverlap_windows.csv
reports/sanity_checks/folder_overlap_summary.csv
reports/sanity_checks/dataset_split_summary.csv
```

For deprecated/original trained models:

```text
models/final_stage2/stage2_activity_imu_single_seed42/
models/final_stage2/stage2_surface_image_audio_concat_seed42/
```
