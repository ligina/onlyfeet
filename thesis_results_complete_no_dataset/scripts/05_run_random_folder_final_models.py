#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run repeated random folder-level split diagnostics for final candidates only."""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd, log_file: Path, dry_run=False):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN]", " ".join(map(str, cmd)))
    if dry_run:
        return
    with log_file.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed with code {p.returncode}. See {log_file}")


def parse_candidate(s: str):
    task, fusion, mods = s.split(":", 2)
    return task.strip(), fusion.strip(), mods.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", required=True, help="Prepared datasets root containing random_folder/*")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--train_script", default="scripts/02_train_task_cv_model.py")
    ap.add_argument("--candidates", nargs="*", default=["activity:single:imu", "surface:concat:image,audio"])
    ap.add_argument("--seed", type=int, default=42, help="Training seed; split seed is encoded in dataset folder name")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--feature_dim", type=int, default=96)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    datasets_root = Path(args.datasets_root)
    random_root = datasets_root / "random_folder"
    folds = sorted([p for p in random_root.glob("random_folder_seed*") if p.is_dir()])
    if not folds and not args.dry_run:
        raise FileNotFoundError(f"No random_folder_seed* prepared datasets under {random_root}")
    candidates = [parse_candidate(x) for x in args.candidates]

    for data_dir in folds:
        fold = data_dir.name
        for task, fusion, mods in candidates:
            run_name = f"{fold}__{task}__{fusion}__{mods.replace(',', '-')}__trainseed{args.seed}"
            out_dir = Path(args.out_root) / "random_folder_final" / run_name
            if args.resume and (out_dir / "summary.json").exists():
                print("[SKIP]", out_dir)
                continue
            cmd = [
                sys.executable, args.train_script,
                "--data_dir", str(data_dir), "--out_dir", str(out_dir), "--fold", fold,
                "--task", task, "--modalities", mods, "--fusion", fusion,
                "--seed", str(args.seed), "--epochs", str(args.epochs), "--batch", str(args.batch),
                "--lr", str(args.lr), "--dropout", str(args.dropout), "--feature_dim", str(args.feature_dim),
                "--label_smoothing", str(args.label_smoothing), "--patience", str(args.patience),
            ]
            run(cmd, out_dir / "run.log", dry_run=args.dry_run)

    print("[DONE] random folder final-model diagnostics")


if __name__ == "__main__":
    main()
