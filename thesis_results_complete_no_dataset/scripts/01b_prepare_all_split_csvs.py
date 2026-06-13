#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build NPZ datasets for every split CSV produced by 00_make_lopgo_and_random_split_csvs.py.

It calls 01_prepare_task_datasets_rgb.py repeatedly. For held-out test folds, the
existing preparation script writes the test group to regular/dataset_val.npz.
Training code in this package treats dataset_val.npz as held-out test data and
creates an internal validation split from dataset_train.npz.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN]", " ".join(map(str, cmd)))
    with log_file.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with code {p.returncode}. See {log_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_root", required=True, help="Directory produced by 00_make_lopgo_and_random_split_csvs.py")
    ap.add_argument("--data_root", required=True, help="Raw OnlyFeet data root")
    ap.add_argument("--out_root", required=True, help="Output root for prepared NPZ datasets")
    ap.add_argument("--prepare_script", default="scripts/01_prepare_task_datasets_rgb.py")
    ap.add_argument("--include_random", action="store_true", help="Also prepare random-folder splits")
    args = ap.parse_args()

    split_root = Path(args.split_root)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    prepare_script = Path(args.prepare_script)

    jobs = []
    for csv_path in sorted((split_root / "lopgo").glob("lopgo_test_*.csv")):
        fold_name = csv_path.stem.replace("lopgo_test_", "test_")
        jobs.append((csv_path, out_root / "lopgo" / fold_name))

    if args.include_random:
        for csv_path in sorted((split_root / "random_folder").glob("random_folder_seed*.csv")):
            jobs.append((csv_path, out_root / "random_folder" / csv_path.stem))

    if not jobs:
        raise RuntimeError(f"No split CSVs found under {split_root}")

    for csv_path, out_dir in jobs:
        log_file = out_dir / "prepare.log"
        if (out_dir / "regular" / "dataset_train.npz").exists() and (out_dir / "regular" / "dataset_val.npz").exists():
            print("[SKIP] already prepared:", out_dir)
            continue
        cmd = [
            sys.executable,
            str(prepare_script),
            "--csv", str(csv_path),
            "--data_root", str(data_root),
            "--out_dir", str(out_dir),
        ]
        run(cmd, log_file)
        print("[OK] prepared", out_dir)

    print("[DONE] Prepared datasets:", out_root)


if __name__ == "__main__":
    main()
