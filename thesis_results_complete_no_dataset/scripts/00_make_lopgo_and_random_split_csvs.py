#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create recoverable participant-group-held-out and random folder-level split CSVs.

This script does not create window tensors. It only rewrites the split column in
labels_all.csv so that the existing dataset preparation script can build train/test
NPZ files. The held-out group is encoded as split='val' because the existing
preparation script writes dataset_val.npz.

Recoverable participant groups:
  - P1/P2 merged group
  - P3
  - P4

Strict 4-fold LOSO is not claimed here because P1 and P2 are assumed to be
irreversibly merged in the available metadata.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def slug(x) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower().replace("\\", "/")
    s = re.sub(r"[^a-z0-9_/.-]+", "_", s)
    return s


def find_folder_column(df: pd.DataFrame) -> str:
    for c in ["relative_path", "folder", "folder_path", "path", "sample_path"]:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find folder/path column. Existing columns: {df.columns.tolist()}")


def find_group_column(df: pd.DataFrame) -> Optional[str]:
    for c in [
        "participant_group", "participant", "subject", "subject_id", "person", "user",
        "participant_id", "p_id", "group", "split_participant"
    ]:
        if c in df.columns:
            return c
    return None


def infer_group_from_text(text: str) -> str:
    s = slug(text)
    compact = s.replace("_", "").replace("-", "").replace("/", "")

    # P1/P2 merged naming variants.
    if any(k in compact for k in ["p1and2", "p1p2", "p12", "p1+p2"]):
        return "P1P2"

    # Direct P3/P4 patterns. Use boundaries to avoid matching data3 etc too aggressively.
    if re.search(r"(^|[^a-z0-9])p3([^a-z0-9]|$)", s) or "participant3" in compact:
        return "P3"
    if re.search(r"(^|[^a-z0-9])p4([^a-z0-9]|$)", s) or "participant4" in compact:
        return "P4"

    # If folder names start with semantic labels like P3_walk_...
    if compact.startswith("p3"):
        return "P3"
    if compact.startswith("p4"):
        return "P4"
    if compact.startswith("p1and2") or compact.startswith("p1p2") or compact.startswith("p12"):
        return "P1P2"

    return "UNKNOWN"


def normalize_group(value: str) -> str:
    s = slug(value)
    compact = s.replace("_", "").replace("-", "").replace("/", "")
    if compact in {"p1and2", "p1p2", "p12", "p1p2merged", "p1and2merged"}:
        return "P1P2"
    if compact in {"p3", "participant3", "subject3"}:
        return "P3"
    if compact in {"p4", "participant4", "subject4"}:
        return "P4"
    inferred = infer_group_from_text(str(value))
    return inferred


def attach_group(df: pd.DataFrame, participant_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    folder_col = find_folder_column(df)

    if participant_col:
        if participant_col not in df.columns:
            raise ValueError(f"participant_col={participant_col} not found in CSV")
        groups = df[participant_col].map(normalize_group)
    else:
        group_col = find_group_column(df)
        if group_col:
            groups = df[group_col].map(normalize_group)
        else:
            groups = df[folder_col].map(infer_group_from_text)

    # Fallback to folder-path inference for rows whose explicit column was unusable.
    bad = groups.eq("UNKNOWN")
    if bad.any():
        groups.loc[bad] = df.loc[bad, folder_col].map(infer_group_from_text)

    df["participant_group_recoverable"] = groups
    unknown = df[df["participant_group_recoverable"].eq("UNKNOWN")]
    if len(unknown):
        sample = unknown[[folder_col]].head(20).to_dict(orient="records")
        raise ValueError(
            f"Could not infer participant group for {len(unknown)} rows. "
            f"Use --participant_col or fix folder names. Examples: {sample}"
        )
    return df


def make_lopgo(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = ["P1P2", "P3", "P4"]
    summary = []
    for g in groups:
        fold = df.copy()
        fold["split"] = np.where(fold["participant_group_recoverable"].eq(g), "val", "train")
        name = f"lopgo_test_{g.lower()}.csv"
        path = out_dir / name
        fold.to_csv(path, index=False)
        summary.append({
            "fold": f"test_{g}",
            "csv": str(path),
            "train_rows": int((fold["split"] == "train").sum()),
            "test_rows_as_val": int((fold["split"] == "val").sum()),
            "train_groups": sorted(fold.loc[fold["split"] == "train", "participant_group_recoverable"].unique().tolist()),
            "test_group": g,
        })
    (out_dir / "lopgo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(out_dir / "lopgo_summary.csv", index=False)


def make_random_folder_splits(df: pd.DataFrame, out_dir: Path, seeds, test_frac: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    folder_col = find_folder_column(df)
    folders = np.array(sorted(df[folder_col].astype(str).unique().tolist()))
    summary = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        shuffled = folders.copy()
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * float(test_frac))))
        test_folders = set(shuffled[:n_test].tolist())
        fold = df.copy()
        fold["split"] = np.where(fold[folder_col].astype(str).isin(test_folders), "val", "train")
        name = f"random_folder_seed{seed}.csv"
        path = out_dir / name
        fold.to_csv(path, index=False)
        summary.append({
            "seed": int(seed),
            "csv": str(path),
            "test_frac": float(test_frac),
            "train_rows": int((fold["split"] == "train").sum()),
            "test_rows_as_val": int((fold["split"] == "val").sum()),
            "train_folders": int(fold.loc[fold["split"] == "train", folder_col].nunique()),
            "test_folders": int(fold.loc[fold["split"] == "val", folder_col].nunique()),
        })
    (out_dir / "random_folder_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(out_dir / "random_folder_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True, help="Original labels_all*.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory for generated split CSVs")
    ap.add_argument("--participant_col", default=None, help="Optional explicit participant/group column name")
    ap.add_argument("--random_seeds", default="42,43,44,45,46")
    ap.add_argument("--random_test_frac", type=float, default=0.20)
    args = ap.parse_args()

    labels_csv = Path(args.labels_csv)
    out_dir = Path(args.out_dir)
    df = pd.read_csv(labels_csv)
    df.columns = [c.strip() for c in df.columns]
    df = attach_group(df, args.participant_col)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "labels_with_recoverable_participant_group.csv", index=False)
    df["participant_group_recoverable"].value_counts().rename_axis("group").reset_index(name="rows").to_csv(
        out_dir / "participant_group_row_counts.csv", index=False
    )

    make_lopgo(df, out_dir / "lopgo")
    seeds = [int(x.strip()) for x in str(args.random_seeds).split(",") if x.strip()]
    make_random_folder_splits(df, out_dir / "random_folder", seeds, args.random_test_frac)

    print("[OK] Split CSVs created:", out_dir)
    print(df["participant_group_recoverable"].value_counts())


if __name__ == "__main__":
    main()
