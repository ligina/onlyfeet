#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from collections import Counter

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


def resolve_folder(data_root, value):
    raw = str(value).strip().replace("\\", "/")
    p = Path(raw)

    if p.is_absolute() and p.exists():
        return p

    candidates = [
        data_root / raw,
        data_root.parent / raw,
        Path(raw),
    ]

    for c in candidates:
        if c.exists():
            return c

    basename = Path(raw).name
    matches = [m for m in data_root.rglob(basename) if m.is_dir()]
    if len(matches) == 1:
        return matches[0]

    return data_root / raw


def scan_folder(folder: Path):
    pkt_files = sort_by_last_int(list(folder.glob("pkt_*.json")))

    pkt_total = len(pkt_files)
    pkt_with_imu = 0
    imu_items_total = 0
    imu_lens = []
    nonempty_examples = []
    empty_examples = []

    for p in pkt_files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
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
            if len(empty_examples) < 3:
                empty_examples.append(p.name)

    return {
        "pkt_total": pkt_total,
        "pkt_with_imu": pkt_with_imu,
        "pkt_empty_imu": pkt_total - pkt_with_imu,
        "imu_items_total": imu_items_total,
        "imu_len_min": min(imu_lens) if imu_lens else 0,
        "imu_len_max": max(imu_lens) if imu_lens else 0,
        "imu_len_first_10": imu_lens[:10],
        "nonempty_examples": nonempty_examples,
        "empty_examples": empty_examples,
    }


def main():
    data_root = Path("data")
    csv_path = Path("data/labels_all_m14.csv")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    folder_col = "relative_path"

    p4walk = df[
        (df["participant"].astype(str).str.strip() == "P4")
        & (df["label_activity"].map(slug) == "walk")
    ].copy()

    rows = []
    by_env = Counter()

    for _, row in p4walk.iterrows():
        folder = resolve_folder(data_root, row[folder_col])
        info = scan_folder(folder)

        item = {
            "env": row["label_env"],
            "folder": str(folder),
            "csv_path": row[folder_col],
            **info,
        }
        rows.append(item)

        if info["imu_items_total"] > 0:
            by_env[(row["label_env"], "has_imu_folder")] += 1
        else:
            by_env[(row["label_env"], "zero_imu_folder")] += 1

        print("\n" + "=" * 100)
        print(row["label_env"], row[folder_col])
        print("folder:", folder)
        print("pkt_total:", info["pkt_total"])
        print("pkt_with_imu:", info["pkt_with_imu"])
        print("pkt_empty_imu:", info["pkt_empty_imu"])
        print("imu_items_total:", info["imu_items_total"])
        print("imu_len_min/max:", info["imu_len_min"], info["imu_len_max"])
        print("imu_len_first_10:", info["imu_len_first_10"])
        print("nonempty_examples:", info["nonempty_examples"])
        print("empty_examples:", info["empty_examples"])

    out = Path("reports_m14_rgb64/debug/p4_walk_all_imu_scan.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")

    print("\n[SUMMARY BY ENV]")
    for k, v in sorted(by_env.items()):
        print(k, "=", v)

    print("\n[OK] saved:", out)


if __name__ == "__main__":
    main()
