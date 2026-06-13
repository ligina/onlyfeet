# OnlyFeet Post-Feedback LOPGO Experiment Package

This pack is designed for the post-feedback experiment upgrade:

- main generalization evaluation: 3-fold leave-one-participant-group-out (LOPGO)
- full model search: 342 trainings = 3 folds × 2 tasks × 57 configurations
- final-candidate stability: extra seeds for selected models
- diagnostics: majority baseline, non-overlapping windows, folder-level vote, zero-input modality diagnostic, optional label-shuffle training
- optional repeated random folder-level split diagnostics

It does **not** claim strict four-fold LOSO, because P1 and P2 are assumed to be irreversibly merged.

## Repository contents

The Git package intentionally keeps lightweight material needed to audit the
experiment:

- `scripts/`: split generation, dataset preparation, training, orchestration,
  and aggregation code.
- `splits_lopgo/`: recoverable participant-group definitions, LOPGO folds, and
  supplementary random-folder split definitions.
- `reports_lopgo_summary/`: complete run index, cross-fold summaries, selected
  model details, stability results, label-shuffle results, and thesis tables.
- `experiments_lopgo/`: per-run configurations, compact summaries, core metric
  JSON files, and small confusion-matrix/per-class CSV files.

The following generated artifacts remain local and are intentionally excluded
from Git: trained `.h5` models, prepared `.npz` datasets, per-window prediction
CSVs, PNG plots, NumPy matrix files, caches, training histories, epoch logs,
model summaries, classification-report duplicates, and runtime logs. They are
not required to inspect the aggregate claims and would add more than 1 GB to
the repository.

## Completed result evidence

The completed aggregate outputs report:

- Activity recognition selected model: gated IMU+audio, 9 runs across three
  recoverable participant-group folds and three seeds, mean test macro-F1
  94.81% with standard deviation 6.58 percentage points.
- Walking-only surface selected model: image+audio concatenation, 9 runs,
  mean test macro-F1 97.26% with standard deviation 1.16 percentage points.
- Mean non-overlap macro-F1 remains 94.79% for activity and 97.26% for surface.
- Label-shuffle mean macro-F1 falls to 42.04% for activity and 22.53% for
  surface.

These values are copied from
`reports_lopgo_summary/thesis_tables/table1_final_selected_models.csv` and
`table3_label_shuffle_sanity.csv`; those CSV files remain the authoritative
machine-readable source.

## Expected project structure on the server

Example:

```text
/workspace/onlyfeet/
  data/
    labels_all_m14_clean.csv
    ... raw recording folders ...
  scripts/
    00_make_lopgo_and_random_split_csvs.py
    01_prepare_task_datasets_rgb.py
    01b_prepare_all_split_csvs.py
    02_train_task_cv_model.py
    03_run_full_lopgo_search.py
    04_run_final_stability_and_diagnostics.py
    05_run_random_folder_final_models.py
    06_aggregate_experiment_results.py
```

Copy all files in `scripts/` from this pack into your project `scripts/` folder.

## Environment

Use a TensorFlow GPU environment. Minimal packages:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib pillow librosa soundfile audioread
```

Recommended runtime style:

```bash
tmux new -s onlyfeet_lopgo
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1
```

## Step 1: Create split CSVs

```bash
cd /workspace/onlyfeet

python scripts/00_make_lopgo_and_random_split_csvs.py \
  --labels_csv data/labels_all_m14_clean.csv \
  --out_dir splits_lopgo \
  --random_seeds 42,43,44,45,46 \
  --random_test_frac 0.20
```

If participant group cannot be inferred automatically, provide the column explicitly:

```bash
python scripts/00_make_lopgo_and_random_split_csvs.py \
  --labels_csv data/labels_all_m14_clean.csv \
  --out_dir splits_lopgo \
  --participant_col participant
```

Output:

```text
splits_lopgo/lopgo/lopgo_test_p1p2.csv
splits_lopgo/lopgo/lopgo_test_p3.csv
splits_lopgo/lopgo/lopgo_test_p4.csv
splits_lopgo/random_folder/random_folder_seed42.csv
...
```

## Step 2: Prepare NPZ datasets for all LOPGO folds

```bash
python scripts/01b_prepare_all_split_csvs.py \
  --split_root splits_lopgo \
  --data_root data \
  --out_root datasets_lopgo_rgb64 \
  --prepare_script scripts/01_prepare_task_datasets_rgb.py
```

To also prepare random-folder splits:

```bash
python scripts/01b_prepare_all_split_csvs.py \
  --split_root splits_lopgo \
  --data_root data \
  --out_root datasets_lopgo_rgb64 \
  --prepare_script scripts/01_prepare_task_datasets_rgb.py \
  --include_random
```

Each fold will contain:

```text
datasets_lopgo_rgb64/lopgo/test_p1p2/regular/dataset_train.npz
datasets_lopgo_rgb64/lopgo/test_p1p2/regular/dataset_val.npz
```

Important: `dataset_val.npz` is the held-out test group for that fold. The training script creates internal validation from `dataset_train.npz` by folder, so test data is not used for checkpoint selection.

## Step 3: Dry-run the full experiment matrix

Before renting a long run, verify commands:

```bash
python scripts/03_run_full_lopgo_search.py \
  --datasets_root datasets_lopgo_rgb64 \
  --out_root experiments_lopgo \
  --train_script scripts/02_train_task_cv_model.py \
  --epochs 80 \
  --batch 32 \
  --seed 42 \
  --dry_run
```

## Step 4: Run the full 342-training search

```bash
nohup python scripts/03_run_full_lopgo_search.py \
  --datasets_root datasets_lopgo_rgb64 \
  --out_root experiments_lopgo \
  --train_script scripts/02_train_task_cv_model.py \
  --epochs 80 \
  --batch 32 \
  --seed 42 \
  --resume \
  > full_lopgo_search.nohup.log 2>&1 &
```

Monitor:

```bash
tail -f full_lopgo_search.nohup.log
find experiments_lopgo/full_lopgo_search -name summary.json | wc -l
```

Expected completed run count: `342`.

## Step 5: Run final-candidate stability and label-shuffle diagnostics

Default candidates:

- Activity: `single:imu`
- Surface: `concat:image,audio`

```bash
nohup python scripts/04_run_final_stability_and_diagnostics.py \
  --datasets_root datasets_lopgo_rgb64 \
  --out_root experiments_lopgo \
  --train_script scripts/02_train_task_cv_model.py \
  --seeds 42,43,44 \
  --epochs 100 \
  --batch 32 \
  --run_label_shuffle \
  --resume \
  > final_stability.nohup.log 2>&1 &
```

This produces normal stability runs and one label-shuffle sanity run per fold/task candidate.

## Step 6: Optional random folder-level diagnostics

Only do this after LOPGO is finished. It is a supplementary within-distribution diagnostic, not participant-independent evidence.

```bash
nohup python scripts/05_run_random_folder_final_models.py \
  --datasets_root datasets_lopgo_rgb64 \
  --out_root experiments_lopgo \
  --train_script scripts/02_train_task_cv_model.py \
  --epochs 80 \
  --batch 32 \
  --resume \
  > random_folder_final.nohup.log 2>&1 &
```

## Step 7: Aggregate results

```bash
python scripts/06_aggregate_experiment_results.py \
  --results_root experiments_lopgo \
  --out_dir reports_lopgo_summary
```

Key outputs:

```text
reports_lopgo_summary/all_runs.csv
reports_lopgo_summary/cv_summary_by_config.csv
reports_lopgo_summary/top10_by_task.csv
reports_lopgo_summary/selected_config_fold_details.csv
reports_lopgo_summary/cv_summary_formatted.csv
reports_lopgo_summary/selected_final_3seed_results.csv
reports_lopgo_summary/label_shuffle_results.csv
reports_lopgo_summary/thesis_tables/
```

## What to use in the thesis

Main evidence:

- `cv_summary_by_config.csv`
- `top10_by_task.csv`
- `selected_config_fold_details.csv`
- `selected_final_3seed_results.csv`
- `thesis_tables/table1_final_selected_models.csv`
- `thesis_tables/table2_per_fold_selected_models.csv`

Diagnostics:

- per-run `majority_baseline_metrics.json`
- per-run `folder_majority_metrics.json`
- per-run `test_nonoverlap_metrics.json`
- per-run `zero_input_metrics.json`
- `label_shuffle_results.csv`
- `thesis_tables/table3_label_shuffle_sanity.csv`

Recommended claim boundary:

> Since P1 and P2 were irreversibly merged in the available metadata, strict four-fold LOSO cross-validation could not be reconstructed. Therefore, the evaluation uses a three-fold leave-one-participant-group-out protocol over the recoverable groups P1/P2, P3, and P4. Repeated random folder-level splits are reported only as supplementary within-distribution diagnostics and are not treated as participant-independent evidence.

## Expected full experiment count

Full model search:

- 5 single-modality configurations
- 26 multimodal combinations × 2 fusion strategies = 52
- 57 configurations per task
- 2 tasks
- 3 folds

Total: 342 training runs.

Final stability:

- 3 folds × 2 final task candidates × 3 seeds = 18 runs
- label shuffle if enabled: 3 folds × 2 candidates = 6 extra runs

Random folder final diagnostics:

- 5 random splits × 2 candidates = 10 runs

Total if all enabled: 342 + 18 + 6 + 10 = 376 runs.
