#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run final-candidate stability and label-shuffle diagnostics.

Default final candidates:
  activity: IMU-only single model
  surface : image+audio concat model

Runs seeds 42,43,44 for normal stability and seed 42 with shuffled training
labels as a leakage sanity check. The training script already produces held-out,
non-overlap, folder-majority, majority-baseline, and zero-input diagnostics.
"""

import argparse
import json
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
    # format: task:fusion:mod1,mod2
    task, fusion, mods = s.split(":", 2)
    return task.strip(), fusion.strip(), mods.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--train_script", default="scripts/02_train_task_cv_model.py")
    ap.add_argument("--folds", default="test_p1p2,test_p3,test_p4")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--candidates", nargs="*", default=["activity:single:imu", "surface:concat:image,audio"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--feature_dim", type=int, default=96)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--run_label_shuffle", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    folds = [x.strip() for x in args.folds.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    candidates = [parse_candidate(x) for x in args.candidates]
    train_script = Path(args.train_script)
    datasets_root = Path(args.datasets_root)
    out_root = Path(args.out_root)
    manifest = []

    for fold in folds:
        data_dir = datasets_root / "lopgo" / fold
        for task, fusion, mods in candidates:
            for seed in seeds:
                run_name = f"{fold}__{task}__{fusion}__{mods.replace(',', '-')}__seed{seed}"
                out_dir = out_root / "final_stability" / run_name
                manifest.append({"fold": fold, "task": task, "fusion": fusion, "modalities": mods, "seed": seed, "out_dir": str(out_dir), "shuffle": False})
                if args.resume and (out_dir / "summary.json").exists():
                    print("[SKIP]", out_dir)
                else:
                    cmd = [
                        sys.executable, str(train_script),
                        "--data_dir", str(data_dir), "--out_dir", str(out_dir), "--fold", fold,
                        "--task", task, "--modalities", mods, "--fusion", fusion,
                        "--seed", str(seed), "--epochs", str(args.epochs), "--batch", str(args.batch),
                        "--lr", str(args.lr), "--dropout", str(args.dropout), "--feature_dim", str(args.feature_dim),
                        "--label_smoothing", str(args.label_smoothing), "--patience", str(args.patience),
                    ]
                    run(cmd, out_dir / "run.log", dry_run=args.dry_run)

            if args.run_label_shuffle:
                seed = seeds[0]
                run_name = f"{fold}__{task}__{fusion}__{mods.replace(',', '-')}__seed{seed}__label_shuffle"
                out_dir = out_root / "label_shuffle" / run_name
                manifest.append({"fold": fold, "task": task, "fusion": fusion, "modalities": mods, "seed": seed, "out_dir": str(out_dir), "shuffle": True})
                if args.resume and (out_dir / "summary.json").exists():
                    print("[SKIP]", out_dir)
                else:
                    cmd = [
                        sys.executable, str(train_script),
                        "--data_dir", str(data_dir), "--out_dir", str(out_dir), "--fold", fold,
                        "--task", task, "--modalities", mods, "--fusion", fusion,
                        "--seed", str(seed), "--epochs", str(args.epochs), "--batch", str(args.batch),
                        "--lr", str(args.lr), "--dropout", str(args.dropout), "--feature_dim", str(args.feature_dim),
                        "--label_smoothing", str(args.label_smoothing), "--patience", str(args.patience),
                        "--shuffle_train_labels",
                    ]
                    run(cmd, out_dir / "run.log", dry_run=args.dry_run)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "final_stability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[DONE]", len(manifest), "runs in manifest")


if __name__ == "__main__":
    main()
