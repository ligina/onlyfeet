#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate OnlyFeet LOPGO experiment outputs into CSV tables for the thesis."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_summary(path: Path):
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    cfg = d.get("config", {})
    row = {
        "run_dir": str(path.parent),
        "fold": cfg.get("fold"),
        "task": cfg.get("task"),
        "fusion": cfg.get("fusion"),
        "modalities": "+".join(cfg.get("modalities", [])),
        "seed": cfg.get("seed"),
        "shuffle_train_labels": cfg.get("shuffle_train_labels", False),
        "n_train_internal": cfg.get("n_train_internal"),
        "n_internal_val": cfg.get("n_internal_val"),
        "n_test_heldout": cfg.get("n_test_heldout"),
        "params": d.get("model_params"),
        "size_mb": d.get("best_model_size_mb"),
    }
    for section in ["test", "nonoverlap", "folder_majority", "majority_baseline"]:
        m = d.get(section, {}) or {}
        for k in ["accuracy", "macro_f1", "weighted_f1", "n"]:
            if k in m:
                row[f"{section}_{k}"] = m[k]
        if "majority_class" in m:
            row["majority_class"] = m["majority_class"]
    zero = d.get("zero_input", {}) or {}
    for cond, m in zero.items():
        if isinstance(m, dict):
            row[f"{cond}_accuracy"] = m.get("accuracy")
            row[f"{cond}_macro_f1"] = m.get("macro_f1")
    return row


def mean_std(x):
    arr = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return ""
    if len(arr) == 1:
        return f"{arr[0]:.4f}"
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    root = Path(args.results_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in root.rglob("summary.json"):
        row = read_summary(p)
        if row:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No summary.json files found under {root}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "all_runs.csv", index=False)

    normal = df[df["shuffle_train_labels"].astype(str).str.lower().isin(["false", "0", "none"])]
    group_cols = ["task", "fusion", "modalities"]
    agg = normal.groupby(group_cols).agg(
        folds=("fold", lambda s: ",".join(sorted(set(map(str, s))))),
        n_runs=("run_dir", "count"),
        mean_test_accuracy=("test_accuracy", "mean"),
        std_test_accuracy=("test_accuracy", "std"),
        mean_test_macro_f1=("test_macro_f1", "mean"),
        std_test_macro_f1=("test_macro_f1", "std"),
        mean_nonoverlap_macro_f1=("nonoverlap_macro_f1", "mean"),
        mean_folder_macro_f1=("folder_majority_macro_f1", "mean"),
        mean_majority_baseline_macro_f1=("majority_baseline_macro_f1", "mean"),
        mean_params=("params", "mean"),
        mean_size_mb=("size_mb", "mean"),
    ).reset_index()
    agg = agg.sort_values(["task", "mean_test_macro_f1", "mean_test_accuracy"], ascending=[True, False, False])
    agg.to_csv(out / "cv_summary_by_config.csv", index=False)

    # Top-10 per task.
    top_rows = []
    for task, sub in agg.groupby("task"):
        top_rows.append(sub.head(10))
    if top_rows:
        pd.concat(top_rows, ignore_index=True).to_csv(out / "top10_by_task.csv", index=False)

    # Thesis-friendly fold table for the best row per task and for common final candidates.
    detailed = normal.copy()
    detailed["config_key"] = detailed["task"] + "__" + detailed["fusion"] + "__" + detailed["modalities"]
    wanted = []
    for task, sub in agg.groupby("task"):
        if len(sub):
            r = sub.iloc[0]
            wanted.append(f"{r['task']}__{r['fusion']}__{r['modalities']}")
    wanted += ["activity__single__imu", "surface__concat__image+audio"]
    wanted = sorted(set(wanted))
    fold_table = detailed[detailed["config_key"].isin(wanted)].sort_values(["task", "config_key", "fold", "seed"])
    fold_table.to_csv(out / "selected_config_fold_details.csv", index=False)

    # Label-shuffle diagnostics.
    shuffle = df[df["shuffle_train_labels"].astype(str).str.lower().isin(["true", "1"])]
    if len(shuffle):
        shuffle.to_csv(out / "label_shuffle_runs.csv", index=False)

    # Add formatted mean±std table.
    fmt_rows = []
    for keys, sub in normal.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(sub)
        row["test_accuracy_mean_std"] = mean_std(sub["test_accuracy"])
        row["test_macro_f1_mean_std"] = mean_std(sub["test_macro_f1"])
        row["nonoverlap_macro_f1_mean_std"] = mean_std(sub.get("nonoverlap_macro_f1", pd.Series(dtype=float)))
        row["folder_macro_f1_mean_std"] = mean_std(sub.get("folder_majority_macro_f1", pd.Series(dtype=float)))
        row["params_mean"] = pd.to_numeric(sub["params"], errors="coerce").mean()
        fmt_rows.append(row)
    fmt = pd.DataFrame(fmt_rows).sort_values(["task", "test_macro_f1_mean_std"], ascending=[True, False])
    fmt.to_csv(out / "cv_summary_formatted.csv", index=False)

    print("[DONE] Aggregated", len(df), "runs")
    print("[OUT]", out)


if __name__ == "__main__":
    main()
