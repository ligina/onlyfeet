#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


def slug(s):
    if s is None:
        return "unknown"
    s = str(s).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s or "unknown"


def find_folder_column(df):
    for c in ["relative_path", "folder", "folder_path", "path", "sample_path", "sample_folder"]:
        if c in df.columns:
            return c
    raise ValueError(df.columns.tolist())


def resolve_folder(data_root, value):
    raw = str(value).strip().replace("\\", "/")
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p

    project_root = data_root.parent
    candidates = []

    if raw.startswith("data/"):
        candidates.append(project_root / raw)

    candidates.extend([
        data_root / raw,
        project_root / raw,
        Path(raw),
    ])

    for c in candidates:
        if c.exists():
            return c

    basename = Path(raw).name
    matches = [m for m in data_root.rglob(basename) if m.is_dir()]
    if len(matches) == 1:
        return matches[0]

    return candidates[0]


def count_files(folder):
    return {
        "exists": folder.exists(),
        "pkt": len(list(folder.glob("pkt_*.json"))) if folder.exists() else 0,
        "wav": len(list(folder.glob("rec_*.wav"))) if folder.exists() else 0,
        "img": len(list(folder.glob("img_*.jpg"))) if folder.exists() else 0,
        "mid": len(list(folder.glob("mid_*.jpg"))) if folder.exists() else 0,
    }


def has_imu_packet(folder):
    pkt_files = sorted(folder.glob("pkt_*.json"))
    for p in pkt_files[:20]:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        imu = obj.get("IMU", obj.get("imu", []))
        if isinstance(imu, list) and len(imu) > 0:
            return True
    return False


def main():
    csv_path = Path("data/labels_all_m14.csv")
    data_root = Path("data")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    folder_col = find_folder_column(df)

    rows = []
    summary = Counter()
    by_pae = Counter()

    for _, row in df.iterrows():
        participant = str(row["participant"]).strip()
        act = slug(row["label_activity"])
        env = str(row["label_env"]).strip()
        folder = resolve_folder(data_root, row[folder_col])
        c = count_files(folder)
        imu_ok = has_imu_packet(folder) if c["exists"] and c["pkt"] > 0 else False

        bad = (
            (not c["exists"])
            or c["pkt"] == 0
            or not imu_ok
        )

        if bad:
            item = {
                "participant": participant,
                "activity": act,
                "env": env,
                "csv_path": str(row[folder_col]),
                "resolved_folder": str(folder),
                **c,
                "imu_packet_detected": imu_ok,
            }
            rows.append(item)
            summary["bad_total"] += 1
            by_pae[(participant, act, env)] += 1

    out = Path("reports_m14_rgb64/debug")
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out / "bad_or_no_imu_folders.csv", index=False, encoding="utf-8")

    print("[BAD SUMMARY]")
    print("bad_total =", summary["bad_total"])

    print("\n[BAD BY participant/activity/env]")
    for k, v in sorted(by_pae.items()):
        print(k, "=", v)

    print("\n[P4 walking rows in CSV]")
    p4walk = df[(df["participant"].astype(str).str.strip() == "P4") & (df["label_activity"].map(slug) == "walk")]
    print(p4walk[["participant", "label_activity", "label_env", folder_col]].to_string(index=False))

    print("\n[OUTPUT]")
    print(out / "bad_or_no_imu_folders.csv")


if __name__ == "__main__":
    main()
