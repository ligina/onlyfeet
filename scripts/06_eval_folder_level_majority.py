#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--pred_npz", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--task", required=True, choices=["activity", "surface"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)
    pred = np.load(args.pred_npz, allow_pickle=True)
    card = json.loads(Path(args.card).read_text(encoding="utf-8"))

    if "folder" not in data.files:
        raise KeyError("The npz file does not contain a 'folder' field.")

    folders = data["folder"].astype(str)

    if args.task == "activity":
        y_all = data["y_act"].astype(int)
        class_names = card["activity_classes"]
    else:
        y_all = data["y_env"].astype(int)
        class_names = card["env_classes"]

    valid = y_all >= 0
    folders = folders[valid]
    y_true_win = y_all[valid]
    y_pred_win = pred["y_pred"].astype(int)

    if len(y_pred_win) != len(y_true_win):
        raise ValueError(f"Length mismatch: y_pred={len(y_pred_win)}, y_true={len(y_true_win)}")

    by_folder = defaultdict(lambda: {"true": [], "pred": []})

    for f, yt, yp in zip(folders, y_true_win, y_pred_win):
        by_folder[f]["true"].append(int(yt))
        by_folder[f]["pred"].append(int(yp))

    rows = []
    y_true_folder = []
    y_pred_folder = []

    for f, d in sorted(by_folder.items()):
        true_counts = Counter(d["true"])
        pred_counts = Counter(d["pred"])

        yt = true_counts.most_common(1)[0][0]
        yp = pred_counts.most_common(1)[0][0]

        y_true_folder.append(yt)
        y_pred_folder.append(yp)

        rows.append({
            "folder": f,
            "n_windows": len(d["true"]),
            "y_true": yt,
            "y_pred": yp,
            "y_true_name": class_names[yt],
            "y_pred_name": class_names[yp],
            "correct": int(yt == yp),
            "window_accuracy_in_folder": sum(int(a == b) for a, b in zip(d["true"], d["pred"])) / len(d["true"]),
            "pred_counts": json.dumps({class_names[k]: v for k, v in pred_counts.items()}, ensure_ascii=False),
        })

    metrics = {
        "task": args.task,
        "n_folders": int(len(y_true_folder)),
        "accuracy": float(accuracy_score(y_true_folder, y_pred_folder)),
        "macro_f1": float(f1_score(y_true_folder, y_pred_folder, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_folder, y_pred_folder, average="weighted", zero_division=0)),
        "classes": class_names,
    }

    pd.DataFrame(rows).to_csv(out_dir / "folder_level_predictions.csv", index=False)
    (out_dir / "folder_level_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = classification_report(
        y_true_folder,
        y_pred_folder,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (out_dir / "folder_level_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true_folder, y_pred_folder, labels=list(range(len(class_names))))
    np.savetxt(out_dir / "folder_level_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print(json.dumps(metrics, indent=2))
    print(report)


if __name__ == "__main__":
    main()
