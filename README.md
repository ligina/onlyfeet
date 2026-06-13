# OnlyFeet Thesis Results

## Overview

This repository contains scripts and lightweight result summaries for the
bachelor thesis **"Multimodal Human Activity Recognition Using a Foot-Mounted
Edge Sensor Platform."**

The final thesis evidence is the three-fold
leave-one-recoverable-participant-group-out (LOPGO) evaluation in
`thesis_results_complete_no_dataset/`. LOPGO is the final thesis protocol.
Earlier P4-only and Clean-P4 artifacts may remain elsewhere in the repository
as historical or traceability context, but they are not the main thesis result.

The project evaluates two tasks using the OnlyFeet foot-mounted sensing
platform:

- Activity recognition from walking, standing, and sitting.
- Walking-only surface recognition for asphalt, PVC, sand, gravel, and grass.

## Final Evaluation Protocol

The final evaluation uses three recoverable participant groups:

- `P1P2`
- `P3`
- `P4`

Each fold holds out one complete recoverable group for testing while training
on the other groups. Early P1 and P2 recordings cannot be separated reliably
in the recoverable metadata, so they are treated as the merged `P1P2` group.
The protocol is therefore three-fold LOPGO, not strict four-subject LOSO.

The full model search used seed 42 across 342 runs:

```text
3 folds x 2 tasks x 57 model configurations
```

The selected final models were then evaluated with seeds 42, 43, and 44 across
all three folds.

## Main Final Results

| Task | Final model | Mean macro-F1 | Key fold-level observation |
| --- | --- | ---: | --- |
| Activity recognition | Gated IMU+audio | 94.81% | P1P2 is hardest at 86.13%; P4 is easiest at 99.89% |
| Walking-only surface recognition | Audio+image concatenation | 97.26% | Fold macro-F1 ranges from 95.81% to 98.00% |

The activity classes are walking, standing, and sitting. The surface classes
are asphalt, PVC, sand, gravel, and grass. Surface recognition is evaluated
only for walking samples; it is not general surface recognition across all
activities.

Authoritative result tables:

```text
thesis_results_complete_no_dataset/reports_lopgo_summary/thesis_tables/table1_final_selected_models.csv
thesis_results_complete_no_dataset/reports_lopgo_summary/thesis_tables/table2_per_fold_selected_models.csv
```

## Sanity Diagnostics

- Label-shuffle sanity check: mean macro-F1 drops by 52.76 percentage points
  for activity and 74.74 percentage points for surface.
- Non-overlap diagnostic: mean macro-F1 is 94.79% for activity and 97.26% for
  surface.
- The non-overlap result reduces concern that direct 50% window overlap is the
  primary driver of the scores. It does not remove participant, session,
  location, or protocol confounds and does not establish real-world
  independence.

The label-shuffle values are recorded in:

```text
thesis_results_complete_no_dataset/reports_lopgo_summary/thesis_tables/table3_label_shuffle_sanity.csv
```

## Repository Contents

The final lightweight evidence bundle is:

```text
thesis_results_complete_no_dataset/
├── experiments_lopgo/
├── reports_lopgo_summary/
│   └── thesis_tables/
├── scripts/
├── splits_lopgo/
└── README_RUNBOOK.md
```

Important included material:

- `thesis_results_complete_no_dataset/reports_lopgo_summary/all_runs.csv`:
  index of completed runs.
- `thesis_results_complete_no_dataset/reports_lopgo_summary/cv_summary_by_config.csv`:
  cross-fold configuration summary, including single-modality and multimodal
  configurations.
- `thesis_results_complete_no_dataset/reports_lopgo_summary/selected_final_3seed_results.csv`:
  selected-model results for seeds 42, 43, and 44.
- `thesis_results_complete_no_dataset/reports_lopgo_summary/label_shuffle_results.csv`:
  label-shuffle runs.
- `thesis_results_complete_no_dataset/reports_lopgo_summary/thesis_tables/`:
  compact thesis-ready final tables.
- `thesis_results_complete_no_dataset/splits_lopgo/lopgo/`: the three LOPGO
  split definitions and summaries.
- `thesis_results_complete_no_dataset/splits_lopgo/random_folder/`:
  supplementary random-folder split definitions.
- `thesis_results_complete_no_dataset/scripts/`: split generation,
  preparation, training, orchestration, and aggregation scripts.
- `thesis_results_complete_no_dataset/experiments_lopgo/`: lightweight
  per-run configs, summaries, JSON metrics, and small CSV reports.
- `thesis_results_complete_no_dataset/README_RUNBOOK.md`: detailed protocol,
  evidence map, and reproduction notes.

## What Is Intentionally Excluded

The following artifacts are intentionally excluded from Git:

- Raw sensor and image/audio data.
- Prepared `.npz` datasets.
- Trained model files such as `.h5` and `.keras`.
- Full per-window prediction CSV files.
- Large runtime and training logs.
- PNG figures, NPY matrices, and other generated heavy artifacts.
- Cache directories and backup files.

These exclusions keep the repository manageable, respect data-handling and
privacy constraints, and emphasize reproducibility through scripts, split
definitions, configurations, and compact result summaries rather than binary
artifacts.

## How to Use This Repository

1. Start with
   `thesis_results_complete_no_dataset/reports_lopgo_summary/thesis_tables/`
   for the final reported values.
2. Inspect
   `thesis_results_complete_no_dataset/reports_lopgo_summary/cv_summary_by_config.csv`,
   `thesis_results_complete_no_dataset/reports_lopgo_summary/selected_config_fold_details.csv`,
   and the per-run JSON metrics for additional traceability.
3. Read
   `thesis_results_complete_no_dataset/README_RUNBOOK.md` for the detailed
   protocol and file map.
4. Use the included scripts as reproducibility references.

A complete rerun is not possible from the Git bundle alone. It requires the
original raw data or prepared datasets and the appropriate local compute
environment.

## Interpretation Boundary

These results are controlled-setting evidence, not proof of open-world
robustness or deployment readiness. Important limitations include fixed
collection locations, a small and relatively homogeneous participant cohort,
the merged `P1P2` group, controlled activity execution, and possible
session- or location-specific visual and acoustic cues.
