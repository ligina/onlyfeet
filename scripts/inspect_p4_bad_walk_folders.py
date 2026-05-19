#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import pandas as pd


def slug(s):
    s = str(s).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def resolve_folder(data_root, value):
    raw = str(value).strip().replace("\\", "/")
    p = Path(raw)

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


def inspect_one(folder):
    info = {}
    info["exists"] = folder.exists()
    if not folder.exists():
        return info

    pkt = sorted(folder.glob("pkt_*.json"))
    wav = sorted(folder.glob("rec_*.wav"))
    img = sorted(folder.glob("img_*.jpg"))
    mid = sorted(folder.glob("mid_*.jpg"))

    info["pkt_count"] = len(pkt)
    info["wav_count"] = len(wav)
    info["img_count"] = len(img)
    info["mid_count"] = len(mid)

    info["first_10_files"] = [p.name for p in sorted(folder.iterdir())[:10]]

    if pkt:
        p = pkt[0]
        info["first_pkt"] = p.name
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            info["first_pkt_keys"] = list(obj.keys())
            imu = obj.get("IMU", obj.get("imu", None))
            tof = obj.get("tof", obj.get("ToF", None))
            mag = obj.get("mag", None)

            info["IMU_type"] = type(imu).__name__
            info["IMU_len"] = len(imu) if isinstance(imu, list) else None
            info["tof_type"] = type(tof).__name__
            info["tof_len"] = len(tof) if isinstance(tof, list) else None
            info["mag_type"] = type(mag).__name__

            if isinstance(imu, list) and len(imu) > 0:
                info["first_IMU_item"] = imu[0]
            if isinstance(tof, list) and len(tof) > 0:
                info["first_tof_item"] = tof[0]
            if isinstance(mag, dict):
                info["mag_item"] = mag

        except Exception as e:
            info["first_pkt_error"] = repr(e)

    # Also check whether JSON files have different names.
    json_files = sorted(folder.glob("*.json"))
    info["json_count_all"] = len(json_files)
    info["first_10_json"] = [p.name for p in json_files[:10]]

    return info


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

    out_rows = []

    for _, row in p4walk.iterrows():
        folder = resolve_folder(data_root, row[folder_col])
        info = inspect_one(folder)

        item = {
            "participant": row["participant"],
            "activity": row["label_activity"],
            "env": row["label_env"],
            "csv_path": row[folder_col],
            "resolved_folder": str(folder),
        }
        item.update(info)
        out_rows.append(item)

        print("\n" + "=" * 100)
        print(row["label_env"], row[folder_col])
        print("resolved:", folder)
        for k, v in item.items():
            print(f"{k}: {v}")

    out = Path("reports_m14_rgb64/debug/p4_walk_folder_inspection.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out, index=False, encoding="utf-8")
    print("\n[OK] saved:", out)


if __name__ == "__main__":
    main()
