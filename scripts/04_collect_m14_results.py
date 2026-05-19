#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect metrics.json files from M14 experiment folders into CSV/JSON tables."""

import argparse
import json
import csv
from pathlib import Path


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        kk = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(flatten(v, kk))
        elif isinstance(v, (list, tuple)):
            out[kk] = ",".join(str(x) for x in v)
        else:
            out[kk] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["reports_m14_rgb64", "models_m14_rgb64"])
    ap.add_argument("--out_csv", default="reports_m14_rgb64/summary_all_metrics.csv")
    ap.add_argument("--out_json", default="reports_m14_rgb64/summary_all_metrics.json")
    args = ap.parse_args()

    rows = []
    seen = set()
    for root_s in args.roots:
        root = Path(root_s)
        if not root.exists():
            continue
        for p in sorted(root.glob("**/metrics.json")):
            if p in seen:
                continue
            seen.add(p)
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                rows.append({"metrics_path": str(p), "error": repr(e)})
                continue
            row = flatten(data)
            row["metrics_path"] = str(p)
            row["run_dir"] = str(p.parent)
            cfg = p.parent / "train_config.json"
            if cfg.exists():
                try:
                    c = json.loads(cfg.read_text(encoding="utf-8"))
                    for key in ["task", "modalities", "modalities_parsed", "fusion", "seed", "train_npz", "eval_npz", "train_samples", "eval_samples"]:
                        if key in c:
                            val = c[key]
                            if isinstance(val, list):
                                val = ",".join(str(x) for x in val)
                            row[f"config.{key}"] = val
                except Exception:
                    pass
            rows.append(row)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    keys = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[DONE] rows={len(rows)}")
    print(out_csv)
    print(out_json)


if __name__ == "__main__":
    main()
