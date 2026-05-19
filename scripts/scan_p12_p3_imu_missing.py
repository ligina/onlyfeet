#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan P1and2 and P3 folders for IMU missingness.

This script checks every folder listed in labels_all_m14.csv for participants
P1and2 and P3. It does not rely on generated npz datasets. It directly opens
pkt_*.json files and counts how many packets contain non-empty IMU lists.

Outputs:
  reports_m14_rgb64/debug/p12_p3_imu_missing_scan.csv
  reports_m14_rgb64/debug/p12_p3_imu_missing_summary.csv
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


def slug(s):
    s = str(s).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def parse_last_int_from_stem(path: Path):
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return None


def sort_by_last_int(files):
    def key(p):
        v = parse_last_int_from_stem(p)
        return v if v is not None else 10**18
    return sorted(files, key=key)


def find_folder_column(df):
    for c in ["relative_path", "folder", "folder_path", "path", "sample_path", "sample_folder"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find folder path column. Columns: {df.columns.tolist()}")


def resolve_folder(data_root: Path, value):
    raw = str(value).strip().replace("\\", "/")
    p = Path(raw)

    if p.is_absolute() and p.exists():
        return p

    candidates = [
        data_root / raw,
        data_root.parent / raw,
        Path(raw),
    ]

    if raw.startswith("data/"):
        candidates.insert(0, data_root.parent / raw)

    for c in candidates:
        if c.exists():
            return c

    basename = Path(raw).name
    if basename:
        matches = [m for m in data_root.rglob(basename) if m.is_dir()]
        if len(matches) == 1:
            return matches[0]

    return data_root / raw


def scan_folder(folder: Path):
    pkt_files = sort_by_last_int(list(folder.glob("pkt_*.json")))

    pkt_total = len(pkt_files)
    pkt_with_imu = 0
    pkt_empty_imu = 0
    pkt_json_error = 0
    imu_items_total = 0
    imu_lens = []
    nonempty_examples = []
    empty_examples = []

    for p in pkt_files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pkt_json_error += 1
            continue

        imu = obj.get("IMU", obj.get("imu", []))
        n = len(imu) if isinstance(imu, list) else 0

        imu_lens.append(n)
        imu_items_total += n

        if n > 0:
            pkt_with_imu += 1
            if len(nonempty_examples) < 3:
                nonempty_examples.append(p.name)
        else:
            pkt_empty_imu += 1
            if len(empty_examples) < 3:
                empty_examples.append(p.name)

    if not folder.exists():
        status = "missing_folder"
    elif pkt_total == 0:
        status = "no_pkt"
    elif pkt_with_imu == 0:
        status = "zero_imu"
    elif pkt_empty_imu > 0:
        status = "partial_imu"
    else:
        status = "has_imu"

    return {
        "exists": folder.exists(),
        "pkt_total": pkt_total,
        "pkt_with_imu": pkt_with_imu,
        "pkt_empty_imu": pkt_empty_imu,
        "pkt_json_error": pkt_json_error,
        "imu_items_total": imu_items_total,
        "imu_len_min": min(imu_lens) if imu_lens else 0,
        "imu_len_max": max(imu_lens) if imu_lens else 0,
        "imu_len_first_10": imu_lens[:10],
        "nonempty_examples": ";".join(nonempty_examples),
        "empty_examples": ";".join(empty_examples),
        "status": status,
    }


def main():
    csv_path = Path("data/labels_all_m14.csv")
    data_root = Path("data")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    folder_col = find_folder_column(df)

    target = df[df["participant"].astype(str).str.strip().isin(["P1and2", "P3"])].copy()

    rows = []
    summary = Counter()
    summary_activity = Counter()
    summary_env = Counter()
    summary_participant_activity_env = Counter()

    for _, row in target.iterrows():
        participant = str(row["participant"]).strip()
        activity = slug(row["label_activity"])
        env = str(row["label_env"]).strip()

        folder = resolve_folder(data_root, row[folder_col])
        info = scan_folder(folder)

        item = {
            "participant": participant,
            "activity": activity,
            "env": env,
            "csv_path": row[folder_col],
            "resolved_folder": str(folder),
            **info,
        }
        rows.append(item)

        status = info["status"]
        summary[(participant, status)] += 1
        summary_activity[(participant, activity, status)] += 1
        summary_env[(participant, env, status)] += 1
        summary_participant_activity_env[(participant, activity, env, status)] += 1

    out_dir = Path("reports_m14_rgb64/debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    df_rows = pd.DataFrame(rows)
    out_csv = out_dir / "p12_p3_imu_missing_scan.csv"
    df_rows.to_csv(out_csv, index=False, encoding="utf-8")

    summary_rows = []

    for (participant, status), count in sorted(summary.items()):
        summary_rows.append({
            "level": "participant",
            "participant": participant,
            "activity": "",
            "env": "",
            "status": status,
            "folder_count": count,
        })

    for (participant, activity, status), count in sorted(summary_activity.items()):
        summary_rows.append({
            "level": "participant_activity",
            "participant": participant,
            "activity": activity,
            "env": "",
            "status": status,
            "folder_count": count,
        })

    for (participant, env, status), count in sorted(summary_env.items()):
        summary_rows.append({
            "level": "participant_env",
            "participant": participant,
            "activity": "",
            "env": env,
            "status": status,
            "folder_count": count,
        })

    for (participant, activity, env, status), count in sorted(summary_participant_activity_env.items()):
        summary_rows.append({
            "level": "participant_activity_env",
            "participant": participant,
            "activity": activity,
            "env": env,
            "status": status,
            "folder_count": count,
        })

    out_summary = out_dir / "p12_p3_imu_missing_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False, encoding="utf-8")

    print("[BASIC SUMMARY: participant/status -> folder count]")
    for k, v in sorted(summary.items()):
        print(k, "=", v)

    print("\n[ACTIVITY SUMMARY: participant/activity/status -> folder count]")
    for k, v in sorted(summary_activity.items()):
        print(k, "=", v)

    print("\n[ZERO IMU OR PARTIAL IMU FOLDERS]")
    bad = df_rows[df_rows["status"].isin(["zero_imu", "partial_imu", "no_pkt", "missing_folder"])]
    if len(bad) == 0:
        print("None")
    else:
        cols = [
            "participant",
            "activity",
            "env",
            "status",
            "pkt_total",
            "pkt_with_imu",
            "pkt_empty_imu",
            "imu_items_total",
            "csv_path",
        ]
        print(bad[cols].to_string(index=False))

    print("\n[OUTPUT]")
    print(out_csv)
    print(out_summary)


if __name__ == "__main__":
    main()
