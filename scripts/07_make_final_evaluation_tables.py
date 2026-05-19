#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import pandas as pd


OUT = Path("reports_m14_rgb64/final_eval")
OUT.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    print(f"[WARN] missing: {p}")
    return pd.DataFrame()


def save(df, name):
    path = OUT / name
    df.to_csv(path, index=False)
    print("[OK]", path, "rows=", len(df))


# 1. Stage 1 all results
stage1 = read_csv_if_exists("reports_m14_rgb64/stage1/summary_stage1_all_114.csv")
if len(stage1):
    cols = [
        "task", "modalities", "fusion", "accuracy", "macro_f1", "weighted_f1",
        "model_params", "model_size_mb", "run_dir"
    ]
    cols = [c for c in cols if c in stage1.columns]

    stage1_sorted = stage1[cols].sort_values(["task", "macro_f1"], ascending=[True, False])
    save(stage1_sorted, "table_stage1_all_114_sorted.csv")

    top20 = stage1_sorted.groupby("task", group_keys=False).head(20)
    save(top20, "table_stage1_top20_by_task.csv")

    best_stage1 = stage1_sorted.groupby("task", group_keys=False).head(1)
    save(best_stage1, "table_stage1_best_by_task.csv")

    fusion_summary = (
        stage1[stage1["fusion"].isin(["concat", "gated"])]
        .groupby(["task", "fusion"])["macro_f1"]
        .agg(["count", "mean", "max", "std"])
        .reset_index()
    )
    save(fusion_summary, "table_stage1_fusion_type_summary.csv")


# 2. Stage 2 candidates
stage2 = read_csv_if_exists("reports_m14_rgb64/stage2/summary_stage2_final_candidates.csv")
if len(stage2):
    cols = [
        "task", "modalities", "fusion", "accuracy", "macro_f1", "weighted_f1",
        "model_params", "model_size_mb", "run_dir"
    ]
    cols = [c for c in cols if c in stage2.columns]
    stage2_sorted = stage2[cols].sort_values(["task", "macro_f1"], ascending=[True, False])
    save(stage2_sorted, "table_stage2_final_candidates_sorted.csv")

    # final selected models
    final_rows = []
    a = stage2[
        (stage2["task"] == "activity") &
        (stage2["modalities"].astype(str).str.contains("imu")) &
        (stage2["fusion"] == "single")
    ]
    if len(a):
        final_rows.append(a.sort_values("macro_f1", ascending=False).iloc[0])

    s = stage2[
        (stage2["task"] == "surface") &
        (stage2["modalities"].astype(str).str.contains("image")) &
        (stage2["modalities"].astype(str).str.contains("audio")) &
        (stage2["fusion"] == "concat")
    ]
    if len(s):
        final_rows.append(s.sort_values("macro_f1", ascending=False).iloc[0])

    if final_rows:
        final_df = pd.DataFrame(final_rows)[cols]
        save(final_df, "table_final_selected_models.csv")


# 3. Robustness trainnorm
rob = read_csv_if_exists("reports_m14_rgb64/robustness_trainnorm/summary_robustness_trainnorm_all.csv")
if len(rob):
    cols = ["experiment", "condition", "accuracy", "macro_f1", "weighted_f1", "n_eval"]
    cols = [c for c in cols if c in rob.columns]
    rob = rob[cols].sort_values(["experiment", "condition"])
    save(rob, "table_robustness_trainnorm.csv")


# 4. Early/mid/late extra
early_late = read_csv_if_exists("reports_m14_rgb64/fusion_strategy_extra/summary_stage2_early_late.csv")
if len(early_late):
    cols = [
        "task", "modalities", "fusion", "accuracy", "macro_f1", "weighted_f1",
        "model_params", "model_size_mb", "run_dir"
    ]
    cols = [c for c in cols if c in early_late.columns]
    early_late = early_late[cols].sort_values(["task", "macro_f1"], ascending=[True, False])
    save(early_late, "table_early_mid_late_extra.csv")


# 5. Folder-level
folder = read_csv_if_exists("reports_m14_rgb64/folder_level/summary_folder_level.csv")
if len(folder):
    cols = ["experiment", "task", "n_folders", "accuracy", "macro_f1", "weighted_f1"]
    cols = [c for c in cols if c in folder.columns]
    folder = folder[cols].sort_values(["task", "experiment"])
    save(folder, "table_folder_level_summary.csv")


print("[DONE] final evaluation tables saved to", OUT)
