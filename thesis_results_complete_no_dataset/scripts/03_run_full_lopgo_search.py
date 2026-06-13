#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the full 3-fold participant-group-held-out model search.

Default matrix:
  3 folds × 2 tasks × (5 single + 26 concat + 26 gated) = 342 trainings.

This script runs sequentially by default. Use one GPU per process. For rented
servers, start this inside tmux/screen or run with nohup.
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ALL_MODALITIES = ["imu", "tof", "mag", "audio", "image"]


def all_configs():
    configs = []
    for m in ALL_MODALITIES:
        configs.append({"modalities": [m], "fusion": "single"})
    for r in range(2, len(ALL_MODALITIES) + 1):
        for combo in itertools.combinations(ALL_MODALITIES, r):
            configs.append({"modalities": list(combo), "fusion": "concat"})
            configs.append({"modalities": list(combo), "fusion": "gated"})
    return configs


def run_one(cmd, log_file: Path, dry_run: bool):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN]", " ".join(map(str, cmd)))
    if dry_run:
        return
    with log_file.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with code {p.returncode}. See {log_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", required=True, help="Prepared datasets root; expects lopgo/test_p1p2, test_p3, test_p4")
    ap.add_argument("--out_root", required=True, help="Output root for models/reports")
    ap.add_argument("--train_script", default="scripts/02_train_task_cv_model.py")
    ap.add_argument("--folds", default="test_p1p2,test_p3,test_p4")
    ap.add_argument("--tasks", default="activity,surface")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--feature_dim", type=int, default=96)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--resume", action="store_true", help="Skip runs whose summary.json exists")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    datasets_root = Path(args.datasets_root)
    out_root = Path(args.out_root)
    train_script = Path(args.train_script)
    folds = [x.strip() for x in args.folds.split(",") if x.strip()]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    configs = all_configs()

    manifest = []
    for fold in folds:
        data_dir = datasets_root / "lopgo" / fold
        if not data_dir.exists() and not args.dry_run:
            raise FileNotFoundError(f"Dataset fold not found: {data_dir}")
        for task in tasks:
            for cfg in configs:
                mods = cfg["modalities"]
                fusion = cfg["fusion"]
                run_name = f"{fold}__{task}__{fusion}__{'-'.join(mods)}__seed{args.seed}"
                out_dir = out_root / "full_lopgo_search" / run_name
                manifest.append({
                    "fold": fold,
                    "task": task,
                    "fusion": fusion,
                    "modalities": mods,
                    "seed": args.seed,
                    "out_dir": str(out_dir),
                })
                if args.resume and (out_dir / "summary.json").exists():
                    print("[SKIP]", out_dir)
                    continue
                cmd = [
                    sys.executable, str(train_script),
                    "--data_dir", str(data_dir),
                    "--out_dir", str(out_dir),
                    "--fold", fold,
                    "--task", task,
                    "--modalities", ",".join(mods),
                    "--fusion", fusion,
                    "--seed", str(args.seed),
                    "--epochs", str(args.epochs),
                    "--batch", str(args.batch),
                    "--lr", str(args.lr),
                    "--dropout", str(args.dropout),
                    "--feature_dim", str(args.feature_dim),
                    "--label_smoothing", str(args.label_smoothing),
                    "--patience", str(args.patience),
                ]
                run_one(cmd, out_dir / "run.log", dry_run=args.dry_run)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "full_lopgo_search_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] scheduled/runs={len(manifest)}")


if __name__ == "__main__":
    main()
