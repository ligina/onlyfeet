#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

OUT = Path("reports_m14_rgb64/sanity_checks")
OUT.mkdir(parents=True, exist_ok=True)

rows = []

paths = [
    ("stage2_activity_train", "datasets_m14_rgb64_stage2/activity/dataset_train.npz"),
    ("stage2_activity_test", "datasets_m14_rgb64_stage2/activity/dataset_test.npz"),
    ("stage2_surface_train", "datasets_m14_rgb64_stage2/surface/dataset_train.npz"),
    ("stage2_surface_test", "datasets_m14_rgb64_stage2/surface/dataset_test.npz"),
]

folder_sets = {}

for name, path in paths:
    d = np.load(path, allow_pickle=True)

    folders = set(map(str, d["folder"])) if "folder" in d.files else set()
    participants = sorted(set(map(str, d["participant"]))) if "participant" in d.files else []

    folder_sets[name] = folders

    y_act = d["y_act"].astype(int) if "y_act" in d.files else None
    y_env = d["y_env"].astype(int) if "y_env" in d.files else None

    row = {
        "split": name,
        "path": path,
        "n_windows": int(len(y_act)) if y_act is not None else None,
        "n_folders": int(len(folders)),
        "participants": ",".join(participants),
    }

    if y_act is not None:
        row["activity_bincount"] = json.dumps(np.bincount(y_act[y_act >= 0]).astype(int).tolist())
    if y_env is not None and (y_env >= 0).any():
        row["surface_bincount"] = json.dumps(np.bincount(y_env[y_env >= 0]).astype(int).tolist())

    rows.append(row)

pd.DataFrame(rows).to_csv(OUT / "dataset_split_summary.csv", index=False)

# Folder overlap checks
overlap_rows = []
keys = list(folder_sets.keys())
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = keys[i], keys[j]
        inter = folder_sets[a] & folder_sets[b]
        overlap_rows.append({
            "split_a": a,
            "split_b": b,
            "overlap_folders": len(inter),
            "examples": json.dumps(sorted(list(inter))[:20], ensure_ascii=False),
        })

pd.DataFrame(overlap_rows).to_csv(OUT / "folder_overlap_summary.csv", index=False)

# Label CSV summary
csv_path = Path("data/labels_all_m14_clean.csv")
if csv_path.exists():
    df = pd.read_csv(csv_path)

    summary = {
        "rows": len(df),
        "columns": list(df.columns),
    }

    for c in ["participant", "split_stage1", "split_stage2", "label_activity", "label_env", "use_activity", "use_surface"]:
        if c in df.columns:
            summary[c] = df[c].value_counts(dropna=False).to_dict()

    (OUT / "label_csv_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if "participant" in df.columns and "split_stage2" in df.columns:
        pd.crosstab(df["participant"], df["split_stage2"]).to_csv(OUT / "participant_x_split_stage2.csv")

print("[DONE] sanity checks saved to", OUT)
