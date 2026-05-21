#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
09_eval_nonoverlap_windows.py

Re-evaluate existing P4 predictions on a non-overlapping subset of windows.

Purpose:
- Original window setting: 2s window, 1s stride => 50% overlap.
- This script keeps only windows with at least --gap_ms distance within each folder.
- It does NOT train or run model inference.
- It only reads dataset_test.npz + eval_predictions.npz and recomputes metrics.

Required files:
- dataset_test.npz with y_act/y_env, folder, start_ms
- eval_predictions.npz with y_pred
- dataset_card.json with class names
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def load_class_names(card, task):
    if task == "activity":
        if "activity_classes" in card:
            return list(card["activity_classes"])
        if "act_classes" in card:
            return list(card["act_classes"])
    else:
        if "env_classes" in card:
            return list(card["env_classes"])
        if "surface_classes" in card:
            return list(card["surface_classes"])
        if "environment_classes" in card:
            return list(card["environment_classes"])

    raise KeyError(f"Cannot find class names for task={task} in dataset_card.json")


def select_nonoverlap_indices(folders, start_ms, gap_ms):
    """
    Select non-overlapping windows within each folder.

    For each folder:
    - sort by start_ms
    - keep the first window
    - keep next window only if start_ms >= last_kept_start + gap_ms

    With 2s windows and 1s stride, gap_ms=2000 keeps non-overlapping windows.
    """
    selected = []

    df = pd.DataFrame({
        "idx": np.arange(len(folders)),
        "folder": folders.astype(str),
        "start_ms": start_ms.astype(float),
    })

    for _, g in df.groupby("folder"):
        g = g.sort_values("start_ms")
        last_start = None

        for _, row in g.iterrows():
            s = float(row["start_ms"])
            if last_start is None or s >= last_start + float(gap_ms):
                selected.append(int(row["idx"]))
                last_start = s

    return np.array(sorted(selected), dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to dataset_test.npz")
    ap.add_argument("--pred_npz", required=True, help="Path to eval_predictions.npz")
    ap.add_argument("--card", required=True, help="Path to dataset_card.json")
    ap.add_argument("--task", required=True, choices=["activity", "surface"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gap_ms", type=float, default=2000.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)
    pred = np.load(args.pred_npz, allow_pickle=True)
    card = json.loads(Path(args.card).read_text(encoding="utf-8"))

    required_data_keys = ["folder", "start_ms"]
    for k in required_data_keys:
        if k not in data.files:
            raise KeyError(f"dataset npz does not contain required key: {k}")

    if "y_pred" not in pred.files:
        raise KeyError("prediction npz does not contain required key: y_pred")

    if args.task == "activity":
        if "y_act" not in data.files:
            raise KeyError("dataset npz does not contain y_act")
        y_all = data["y_act"].astype(int)
    else:
        if "y_env" not in data.files:
            raise KeyError("dataset npz does not contain y_env")
        y_all = data["y_env"].astype(int)

    valid = y_all >= 0

    y_true = y_all[valid]
    y_pred = pred["y_pred"].astype(int)

    folders = data["folder"].astype(str)[valid]
    start_ms = data["start_ms"].astype(float)[valid]

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"Length mismatch: len(y_pred)={len(y_pred)}, len(y_true)={len(y_true)}. "
            "Check whether eval_predictions.npz corresponds to this dataset_test.npz."
        )

    selected = select_nonoverlap_indices(folders, start_ms, args.gap_ms)

    y_true_sel = y_true[selected]
    y_pred_sel = y_pred[selected]
    folders_sel = folders[selected]
    start_sel = start_ms[selected]

    class_names = load_class_names(card, args.task)

    acc = accuracy_score(y_true_sel, y_pred_sel)
    macro_f1 = f1_score(y_true_sel, y_pred_sel, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true_sel, y_pred_sel, average="weighted", zero_division=0)

    metrics = {
        "task": args.task,
        "gap_ms": float(args.gap_ms),
        "n_original_windows": int(len(y_true)),
        "n_nonoverlap_windows": int(len(y_true_sel)),
        "n_folders": int(len(set(folders_sel))),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classes": class_names,
    }

    rows = []
    for original_i, yt, yp, folder, st in zip(selected, y_true_sel, y_pred_sel, folders_sel, start_sel):
        rows.append({
            "original_index": int(original_i),
            "folder": str(folder),
            "start_ms": float(st),
            "y_true": int(yt),
            "y_pred": int(yp),
            "y_true_name": class_names[int(yt)],
            "y_pred_name": class_names[int(yp)],
            "correct": int(yt == yp),
        })

    pd.DataFrame(rows).to_csv(out_dir / "nonoverlap_predictions.csv", index=False)

    (out_dir / "nonoverlap_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = classification_report(
        y_true_sel,
        y_pred_sel,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (out_dir / "nonoverlap_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true_sel, y_pred_sel, labels=list(range(len(class_names))))
    np.savetxt(out_dir / "nonoverlap_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print("[DONE]", out_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(report)


if __name__ == "__main__":
    main()
