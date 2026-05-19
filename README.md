下面是完整详细版 `README.md`。你可以直接复制到 GitLab 的 `README.md` 里。内容已经按照你当前 release 文件夹结构写好，并且对应你现在实际包含的文件：scripts、reports、models、logs、environment、data_docs 等。你的 manifest 里也确实包含了最终模型、final_eval 表格、robustness、folder-level、sanity checks、Stage 1/Stage 2 summaries 和核心脚本。

---

# OnlyFeet M14 Final Experiments

This repository contains the final experiment release for the **OnlyFeet M14 bachelor thesis project**. It includes training scripts, evaluation scripts, result reports, selected trained models, logs, dataset/split documentation, and environment information.

The project investigates **multimodal human activity recognition (HAR)** and **surface/environment recognition** using a **foot-mounted sensor platform**. The available sensing modalities are:

* IMU
* RGB image
* audio
* Time-of-Flight (ToF)
* magnetometer

The main goal is to evaluate how different modalities and fusion strategies contribute to activity and surface recognition, with a focus on model performance, robustness, and deployment complexity.

---

## 1. Repository Structure

```text
onlyfeet_m14_gitlab_release/
├── archive_manifest.txt
├── data_docs/
├── environment/
├── logs/
├── models/
├── reports/
└── scripts/
```

### Main directories

```text
scripts/
```

Contains dataset preparation, model training, evaluation, result collection, robustness testing, folder-level evaluation, and sanity-check scripts.

```text
reports/
```

Contains final result tables, Stage 1 summaries, Stage 2 summaries, robustness reports, folder-level evaluation, fusion-strategy comparison, and split sanity checks.

```text
models/
```

Contains the selected final trained Stage 2 models:

* final activity model: IMU-only
* final surface model: Image+Audio mid-level concat fusion

```text
logs/
```

Contains key training and evaluation logs for Stage 2, robustness evaluation, and early/mid/late fusion comparison.

```text
data_docs/
```

Contains dataset and split documentation, including participant definitions and split rules.

```text
environment/
```

Contains environment information, including Python version, installed packages, GPU information, and storage information.

---

## 2. Project Goal

The goal of this project is to design and evaluate a multimodal recognition system using the OnlyFeet foot-mounted sensor platform.

The system targets two main tasks:

1. **Regular Activity Recognition**

   * walk
   * standing
   * sitting

2. **Surface Recognition**

   * asphalt
   * PVC
   * sand
   * gravel
   * grass

Surface recognition is based on **walking-only samples**, because foot-ground interaction cues are primarily available during walking.

---

## 3. Dataset Summary

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

The generated window datasets contain:

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

---

## 4. Participant Definition

The participant labels are defined as follows.

### P1/P2

`P1/P2` is a merged participant group from early recordings collected by Ziang and another student.

During early data collection, there were some recording issues, including battery depletion and partially missing recordings. Because of this, some recordings could no longer be reliably separated into P1 or P2 during later dataset cleanup. Therefore, these recordings are merged as one group: `P1/P2`.

### P3

`P3` is a separate participant recorded on a different date.

### P4

`P4` is another separate participant recorded on a different date. P4 is used as the final held-out participant test set in Stage 2.

---

## 5. Experimental Design

The experiments are organized into two stages.

---

## 5.1 Stage 1: Model Selection

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

Stage 1 is the only stage used for systematic model search.

---

## 5.2 Stage 2: Final Held-Out Evaluation

Stage 2 is used for final held-out evaluation.

```text
Train: P1/P2 + P3
Test: P4
```

In Stage 2, only 12 pre-selected candidate models were evaluated on P4.

P4 was **not** used for full model search. This avoids turning P4 into a second validation set.

### Stage 2 dataset sizes

Activity recognition:

```text
Train: 29,788 windows, 1,058 folders
Test: 2,735 windows, 105 folders
```

Surface recognition:

```text
Train: 10,299 windows, 371 folders
Test: 951 windows, 36 folders
```

---

## 6. Final Selected Models

---

## 6.1 Final Activity Model

The final activity model is:

```text
IMU-only
```

This model was selected because several multimodal models also achieved 100% on the P4 held-out test set, but none improved over IMU-only. Therefore, the lightweight IMU-only model is preferred for wearable and edge deployment.

### P4 window-level result

```text
Accuracy: 100.00%
Macro-F1: 100.00%
Weighted-F1: 100.00%
```

### P4 folder-level majority-vote result

```text
Accuracy: 100.00%
Macro-F1: 100.00%
Weighted-F1: 100.00%
Folders: 105
```

### Complexity

```text
Parameters: 99,139
Model size: 1.22 MB
```

---

## 6.2 Final Surface Model

The final surface model is:

```text
Image + Audio mid-level concat fusion
```

This model was selected because it achieved the best P4 held-out performance among the Stage 2 final candidate models.

### P4 window-level result

```text
Accuracy: 99.26%
Macro-F1: 99.26%
Weighted-F1: 99.26%
```

### P4 folder-level majority-vote result

```text
Accuracy: 100.00%
Macro-F1: 100.00%
Weighted-F1: 100.00%
Folders: 36
```

### Complexity

```text
Parameters: 280,613
Model size: 3.38 MB
```

---

## 7. Main Result Tables

The main final evaluation tables are located in:

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

---

## 8. Stage 1 Summary

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

---

## 9. Stage 2 Summary

Stage 2 contains the final held-out P4 candidate evaluation.

Location:

```text
reports/stage2/
```

Important files:

```text
summary_stage2_final_candidates.csv
summary_stage2_final_candidates.json
```

Stage 2 evaluates 12 candidate models selected after Stage 1. It is not a full model search.

---

## 10. Robustness Evaluation

Missing-modality robustness was evaluated using train-normalized evaluation.

Location:

```text
reports/robustness_trainnorm/
```

Important summary file:

```text
reports/robustness_trainnorm/summary_robustness_trainnorm_all.csv
```

---

## 10.1 Activity Robustness

A diagnostic multimodal activity fusion model was evaluated under missing-modality conditions.

Model:

```text
IMU + Image + Audio + ToF concat
```

Results:

```text
normal:   100.00% macro-F1
no_imu:    16.66% macro-F1
no_image: 100.00% macro-F1
no_tof:   100.00% macro-F1
no_audio:  68.22% macro-F1
```

Interpretation:

Activity recognition is mainly driven by IMU motion cues. Removing IMU causes severe degradation.

---

## 10.2 Surface Robustness

The final Image+Audio surface model was evaluated under missing-modality conditions.

Model:

```text
Image + Audio concat
```

Results:

```text
normal:   99.26% macro-F1
no_image: 47.62% macro-F1
no_audio:  8.79% macro-F1
```

Interpretation:

Surface recognition relies strongly on the audio cue, while image provides complementary terrain context.

---

## 11. Fusion Strategy Comparison

Early fusion, mid-level fusion, and late fusion were compared.

Location:

```text
reports/fusion_strategy_extra/
```

Important files:

```text
summary_stage2_early_late.csv
summary_stage2_early_late.json
```

---

## 11.1 Activity Fusion Strategy

Task:

```text
Activity recognition
```

Modalities:

```text
IMU + Image + Audio + ToF
```

Results:

```text
early-flat fusion: 100.00% macro-F1
mid-level concat:  100.00% macro-F1
late-average:       98.19% macro-F1
```

Observation:

Although early-flat fusion reaches 100%, it is much larger than the selected IMU-only activity model and is not suitable as the final deployment model.

---

## 11.2 Surface Fusion Strategy

Task:

```text
Surface recognition
```

Modalities:

```text
Image + Audio
```

Results:

```text
early-flat fusion: 75.59% macro-F1
mid-level concat:  99.26% macro-F1
late-average:      96.92% macro-F1
```

Interpretation:

Surface recognition is not solved well by simple raw flattening and concatenation. Learning modality-specific representations first and then fusing them at the feature level is more effective.

---

## 12. Folder-Level Majority-Vote Evaluation

Folder-level majority-vote evaluation was added because window-level samples from the same recording are temporally correlated.

Location:

```text
reports/folder_level/
```

Summary file:

```text
reports/folder_level/summary_folder_level.csv
```

Results:

```text
Activity IMU-only:
105 P4 folders
Accuracy: 100.00%
Macro-F1: 100.00%

Surface Image+Audio:
36 P4 folders
Accuracy: 100.00%
Macro-F1: 100.00%
```

Interpretation:

The high P4 results are not only caused by isolated window-level predictions. The predictions are also stable after aggregation at the recording/folder level.

---

## 13. Split Sanity Checks

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
Stage 2 surface train/test folder overlap: 0
```

The activity and surface datasets overlap within the same split because surface recognition is constructed as the walking-only subset of the activity dataset. This is expected and is not train/test leakage.

---

## 14. Trained Models

The selected final Stage 2 models are stored in:

```text
models/final_stage2/
```

Included final models:

```text
models/final_stage2/stage2_activity_imu_single_seed42/
models/final_stage2/stage2_surface_image_audio_concat_seed42/
```

Each model directory contains:

```text
best_model.h5
final_model.h5
metrics.json
train_config.json
history.json
training_log.csv
eval_metrics.json
eval_predictions.csv
eval_predictions.npz
eval_classification_report.txt
eval_confusion_matrix.csv
eval_confusion_matrix.png
eval_confusion_matrix_normalized.png
```

The two final selected models are:

```text
Activity:
stage2_activity_imu_single_seed42

Surface:
stage2_surface_image_audio_concat_seed42
```

---

## 15. Logs

Logs are stored in:

```text
logs/
```

Subdirectories:

```text
logs/stage2/
logs/robustness_trainnorm/
logs/fusion_strategy/
```

These logs document:

* Stage 2 final-candidate training
* robustness evaluation
* early/mid/late fusion comparison

---

## 16. Scripts

The main scripts are located in:

```text
scripts/
```

---

## 16.1 Dataset Preparation

```text
scripts/01_prepare_task_datasets_m14_rgb64.py
```

Builds Stage 1 and Stage 2 RGB64 task datasets from labeled raw folders.

---

## 16.2 Main Training

```text
scripts/02_train_m14_task_model.py
```

Main training script for single-modality, concat fusion, and gated fusion models.

---

## 16.3 Early/Late Fusion Training

```text
scripts/02b_train_m14_early_late_fusion.py
```

Extra script for early-flat and late-average fusion experiments.

---

## 16.4 Robustness Evaluation

```text
scripts/03_eval_m14_model_robustness.py
scripts/03b_eval_m14_model_robustness_trainnorm.py
```

The `03b` version uses train-set normalization and is the final version used for the reported robustness tables.

---

## 16.5 Result Collection

```text
scripts/04_collect_m14_results.py
```

Collects model metrics into CSV/JSON summary tables.

---

## 16.6 Folder-Level Evaluation

```text
scripts/06_eval_folder_level_majority.py
```

Performs folder-level majority-vote evaluation from window-level predictions.

---

## 16.7 Final Table Generation

```text
scripts/07_make_final_evaluation_tables.py
```

Generates final thesis-ready evaluation tables.

---

## 16.8 Sanity Checks

```text
scripts/08_sanity_check_m14_final.py
```

Checks dataset split summaries and folder overlap.

---

## 16.9 Batch Runner Scripts

```text
scripts/run_10_stage1_single_modality.sh
scripts/run_20_stage1_full_fusion.sh
scripts/run_22_stage1_fusion_range.sh
scripts/run_23_one_fusion_job.sh
scripts/run_24_fusion_jobs_loop.sh
scripts/run_31_stage2_final_candidates.sh
scripts/run_40_robustness_template.sh
```

These scripts were used to run Stage 1 model search, Stage 2 final candidates, and robustness evaluations.

---

## 16.10 Data Inspection Scripts

```text
scripts/check_m14_no_window_folders.py
scripts/inspect_p4_bad_walk_folders.py
scripts/scan_p12_p3_imu_missing.py
scripts/scan_p4_all_activity_imu.py
scripts/scan_p4_all_imu_packets.py
```

These scripts were used for data quality inspection and debugging.

---

## 17. Environment

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

## 18. Reproducing the Experiments

The raw dataset is not included in this GitLab release due to size and privacy/storage constraints.

However, the included scripts and reports allow review of:

* split logic
* training setup
* evaluation setup
* model summaries
* logs
* trained final models
* final metrics
* sanity checks

A typical full reproduction would require the original raw dataset or prepared `.npz` datasets.

---

## 18.1 Stage 1 Single-Modality Baselines

```bash
bash scripts/run_10_stage1_single_modality.sh
```

---

## 18.2 Stage 1 Full Fusion Search

```bash
bash scripts/run_20_stage1_full_fusion.sh
```

For more stable execution on limited GPU memory, jobs can be executed sequentially:

```bash
bash scripts/run_24_fusion_jobs_loop.sh 1 104
```

---

## 18.3 Stage 2 Final Candidate Evaluation

```bash
bash scripts/run_31_stage2_final_candidates.sh
```

---

## 18.4 Robustness Evaluation

The final robustness results are generated with:

```bash
python scripts/03b_eval_m14_model_robustness_trainnorm.py
```

The exact commands are documented in logs and runner scripts.

---

## 18.5 Final Tables

```bash
python scripts/07_make_final_evaluation_tables.py
```

---

## 18.6 Sanity Checks

```bash
python scripts/08_sanity_check_m14_final.py
```

---

## 19. Important Caveats and Limitations

The final P4 results are very high:

```text
Activity: 100.00% macro-F1
Surface: 99.26% macro-F1
```

Sanity checks found no train/test folder overlap. However, the data was collected in a relatively controlled setting.

Potential limitations:

1. The number of participants is limited.
2. P1/P2 is a merged group rather than two fully separable participants.
3. Window-level samples from the same recording are temporally correlated.
4. The P4 data may be more standardized than fully unconstrained real-world data.
5. Participants may have performed actions more carefully and consistently during data collection.
6. The results should therefore be interpreted as controlled subject-held-out performance rather than full real-world generalization.

For this reason, both window-level and folder-level metrics are reported.

---

## 20. Recommended Interpretation

The current results support the following conclusions:

1. **Activity recognition is IMU-dominant.**
   A lightweight IMU-only model is sufficient for walk/standing/sitting recognition in the current controlled dataset.

2. **Surface recognition benefits from multimodal fusion.**
   Image+Audio mid-level fusion performs best among the final Stage 2 candidate models.

3. **Mid-level fusion is more effective than early-flat fusion for surface recognition.**
   Early-flat Image+Audio fusion performs much worse than mid-level feature fusion.

4. **Robustness analysis confirms modality importance.**
   Removing IMU strongly damages activity recognition, while removing audio strongly damages surface recognition.

5. **The final results should be reported with caution.**
   The evaluation is subject-held-out and folder-level checked, but the dataset remains controlled and limited in participant diversity.

---

## 21. Key Files for Review

For a quick review, start with:

```text
reports/final_eval/table_final_selected_models.csv
reports/final_eval/table_stage2_final_candidates_sorted.csv
reports/final_eval/table_robustness_trainnorm.csv
reports/final_eval/table_early_mid_late_extra.csv
reports/final_eval/table_folder_level_summary.csv
reports/sanity_checks/folder_overlap_summary.csv
reports/sanity_checks/dataset_split_summary.csv
```

For final trained models:

```text
models/final_stage2/stage2_activity_imu_single_seed42/
models/final_stage2/stage2_surface_image_audio_concat_seed42/
```

For core scripts:

```text
scripts/01_prepare_task_datasets_m14_rgb64.py
scripts/02_train_m14_task_model.py
scripts/03b_eval_m14_model_robustness_trainnorm.py
scripts/06_eval_folder_level_majority.py
scripts/07_make_final_evaluation_tables.py
scripts/08_sanity_check_m14_final.py
```

---

## 22. Contact

Author:

```text
Ziang Liu
Bachelor thesis project: OnlyFeet multimodal HAR and surface recognition
```
