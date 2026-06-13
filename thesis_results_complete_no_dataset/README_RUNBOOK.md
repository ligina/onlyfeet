# Thesis Results Complete - No Dataset

## Purpose

This folder is the lightweight final evidence bundle for the OnlyFeet
three-fold LOPGO thesis evaluation. It supports inspection and traceability
without including raw data, prepared datasets, trained models, predictions, or
other large generated artifacts.

The final thesis protocol and results are represented by the files in this
folder. Earlier P4-only and Clean-P4 artifacts elsewhere in the repository are
historical or traceability material, not the final thesis result.

## Protocol Summary

The evaluation uses three recoverable participant groups:

- `P1P2`
- `P3`
- `P4`

One group is held out in each fold. P1 and P2 cannot be separated reliably in
the recoverable early-recording metadata, so `P1P2` is treated as one group.
This is three-fold leave-one-recoverable-participant-group-out evaluation, not
strict four-subject LOSO.

The experiment sequence was:

1. A seed-42 full model search.
2. Three folds x two tasks x 57 configurations, for 342 search runs.
3. Selection of the final task-specific models.
4. Final stability evaluation with seeds 42, 43, and 44 across all three folds.
5. Supporting label-shuffle, non-overlap, folder-vote, majority-baseline, and
   zero-input diagnostics.

## Final Selected Models

| Task | Selected model | Mean macro-F1 | Mean accuracy | Notes |
| --- | --- | ---: | ---: | --- |
| Activity recognition | Gated IMU+audio | 94.81% | 94.88% | Three folds and three seeds |
| Walking-only surface recognition | Audio+image concatenation | 97.26% | 97.25% | Walking samples only; three folds and three seeds |

The authoritative aggregate table is:

```text
reports_lopgo_summary/thesis_tables/table1_final_selected_models.csv
```

## Fold-Level Results

| Task | Held-out group | Mean macro-F1 |
| --- | --- | ---: |
| Activity recognition | P1P2 | 86.13% |
| Activity recognition | P3 | 98.39% |
| Activity recognition | P4 | 99.89% |
| Walking-only surface recognition | P1P2 | 98.00% |
| Walking-only surface recognition | P3 | 95.81% |
| Walking-only surface recognition | P4 | 97.98% |

Detailed fold-level accuracy, macro-F1, ranges, and non-overlap values are in:

```text
reports_lopgo_summary/thesis_tables/table2_per_fold_selected_models.csv
reports_lopgo_summary/selected_final_3seed_results.csv
```

The activity result varies substantially by held-out group, with P1P2 being
the most difficult fold. The surface result is more consistent across folds.

## Diagnostics

### Label Shuffle

The label-shuffle sanity check reduces mean macro-F1 from 94.81% to 42.04% for
activity and from 97.26% to 22.53% for surface. The corresponding drops are
52.76 and 74.74 percentage points.

```text
reports_lopgo_summary/label_shuffle_results.csv
reports_lopgo_summary/thesis_tables/table3_label_shuffle_sanity.csv
experiments_lopgo/label_shuffle/
```

This check argues against obvious train-label leakage. It does not by itself
prove the absence of every possible confound.

### Non-Overlap

The selected models retain mean non-overlap macro-F1 values of 94.79% for
activity and 97.26% for surface.

Per-run values are stored in `test_nonoverlap_metrics.json`, with aggregate
values in the thesis tables. The diagnostic reduces direct 50% window-overlap
concerns but does not prove participant-, session-, location-, protocol-, or
real-world independence.

### Single-Modality Results

Single-modality configurations are included in the full search summaries:

```text
reports_lopgo_summary/cv_summary_by_config.csv
reports_lopgo_summary/top10_by_task.csv
reports_lopgo_summary/all_runs.csv
```

They are comparison baselines within the same controlled protocol, not
independent deployment studies.

### Zero-Input Perturbation

Available runs include `zero_input_metrics.json` for modality perturbation
inspection. Zero-input perturbation diagnostics are not retrained ablations
and are not causal estimates of modality importance. They show model behavior
under artificial input replacement only.

Additional per-run diagnostics include:

```text
majority_baseline_metrics.json
folder_majority_metrics.json
test_metrics.json
test_nonoverlap_metrics.json
test_per_class.csv
test_nonoverlap_per_class.csv
```

## Included Files

### Aggregate Summaries

```text
reports_lopgo_summary/all_runs.csv
reports_lopgo_summary/cv_summary_by_config.csv
reports_lopgo_summary/cv_summary_formatted.csv
reports_lopgo_summary/final_stability_only.csv
reports_lopgo_summary/label_shuffle_results.csv
reports_lopgo_summary/selected_config_fold_details.csv
reports_lopgo_summary/selected_final_3seed_results.csv
reports_lopgo_summary/top10_by_task.csv
```

### Thesis Tables

```text
reports_lopgo_summary/thesis_tables/table1_final_selected_models.csv
reports_lopgo_summary/thesis_tables/table2_per_fold_selected_models.csv
reports_lopgo_summary/thesis_tables/table3_label_shuffle_sanity.csv
```

### Split Definitions

```text
splits_lopgo/labels_with_recoverable_participant_group.csv
splits_lopgo/participant_group_row_counts.csv
splits_lopgo/lopgo/lopgo_summary.csv
splits_lopgo/lopgo/lopgo_summary.json
splits_lopgo/lopgo/lopgo_test_p1p2.csv
splits_lopgo/lopgo/lopgo_test_p3.csv
splits_lopgo/lopgo/lopgo_test_p4.csv
splits_lopgo/random_folder/random_folder_seed42.csv
splits_lopgo/random_folder/random_folder_seed43.csv
splits_lopgo/random_folder/random_folder_seed44.csv
splits_lopgo/random_folder/random_folder_seed45.csv
splits_lopgo/random_folder/random_folder_seed46.csv
splits_lopgo/random_folder/random_folder_summary.csv
splits_lopgo/random_folder/random_folder_summary.json
```

The random-folder splits are supplementary within-distribution diagnostics.
They are not participant-independent evidence.

### Scripts

```text
scripts/00_make_lopgo_and_random_split_csvs.py
scripts/01_prepare_task_datasets_rgb.py
scripts/01b_prepare_all_split_csvs.py
scripts/02_train_task_cv_model.py
scripts/02_train_task_cv_model_v2.py
scripts/03_run_full_lopgo_search.py
scripts/04_run_final_stability_and_diagnostics.py
scripts/05_run_random_folder_final_models.py
scripts/06_aggregate_experiment_results.py
```

### Per-Run Evidence

The `experiments_lopgo/` tree retains lightweight configurations, summaries,
JSON metrics, and small CSV reports for:

```text
experiments_lopgo/full_lopgo_search/
experiments_lopgo/final_stability/
experiments_lopgo/label_shuffle/
experiments_lopgo/full_lopgo_search_manifest.json
```

## Excluded Files

Repository `.gitignore` rules intentionally exclude:

- Raw data and prepared `.npz` datasets.
- Trained `.h5` and `.keras` models.
- Full `*predictions.csv` files.
- Runtime and training `.log` files.
- PNG plots and NPY matrices.
- Training histories and epoch-level training logs.
- Model summaries and duplicate classification reports.
- Python caches, notebook checkpoints, and script backup files.
- Archives and local virtual environments.

These artifacts remain local where available and are not deleted by the ignore
policy.

## Reproduction Notes

This bundle supports:

- Inspection of final and fold-level result tables.
- Verification of split definitions.
- Review of model configurations and metric JSON files.
- Inspection of the scripts used for preparation, training, orchestration, and
  aggregation.

The scripts document the expected workflow, but a full rerun requires the
original raw recordings or prepared `.npz` datasets, the required software
environment, and sufficient compute resources. The Git bundle alone is not a
self-contained training dataset or model release.
