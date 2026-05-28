#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect Clean-P4 model complexity summary.

This script does not train or evaluate models. It only loads the final Clean-P4
activity model, surface model, and optional composite wrapper, then records:
- parameter counts
- model file size
- input/output names and shapes
- existing evaluation metrics from composite evaluation_summary.json if available

It includes LegacyGRU compatibility for older .h5 models saved with time_major.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="OnlyFeetCompat", name="LegacyGRU")
class LegacyGRU(keras.layers.GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)


LEGACY_CUSTOM_OBJECTS = {
    "GRU": LegacyGRU,
    "keras.layers.GRU": LegacyGRU,
    "tensorflow.keras.layers.GRU": LegacyGRU,
    "LegacyGRU": LegacyGRU,
    "OnlyFeetCompat>LegacyGRU": LegacyGRU,
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_shape(x: Any) -> str:
    try:
        return str(tuple(x.shape))
    except Exception:
        return ""


def count_weights(weights) -> int:
    total = 0
    for w in weights:
        try:
            total += int(tf.keras.backend.count_params(w))
        except Exception:
            pass
    return total


def load_model_legacy(path: Path):
    return keras.models.load_model(
        path,
        compile=False,
        custom_objects=LEGACY_CUSTOM_OBJECTS,
        safe_mode=False,
    )


def extract_branch_metrics(evaluation_summary: dict[str, Any], branch: str) -> dict[str, Any]:
    node = evaluation_summary.get(branch, {})
    metrics = node.get("metrics", {}) if isinstance(node, dict) else {}
    return {
        "accuracy": metrics.get("accuracy", ""),
        "macro_f1": metrics.get("macro_f1", ""),
        "weighted_f1": metrics.get("weighted_f1", ""),
        "n_eval": metrics.get("n_eval", ""),
        "original_vs_composite_max_abs_diff": metrics.get("original_vs_composite_max_abs_diff", ""),
    }


def describe_model(
    name: str,
    path: Path,
    task: str,
    modalities: list[str],
    branch_metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "task": task,
        "modalities": "+".join(modalities),
        "model_path": str(path),
        "exists": bool(path.exists()),
        "model_size_mb": "",
        "params": "",
        "trainable_params": "",
        "non_trainable_params": "",
        "input_names": "",
        "input_shapes": "",
        "output_names": "",
        "output_shapes": "",
        "accuracy": branch_metrics.get("accuracy", ""),
        "macro_f1": branch_metrics.get("macro_f1", ""),
        "weighted_f1": branch_metrics.get("weighted_f1", ""),
        "n_eval": branch_metrics.get("n_eval", ""),
        "original_vs_composite_max_abs_diff": branch_metrics.get("original_vs_composite_max_abs_diff", ""),
        "error": "",
    }

    if not path.exists():
        row["error"] = "model file not found"
        return row

    row["model_size_mb"] = round(path.stat().st_size / 1024 / 1024, 4)

    try:
        model = load_model_legacy(path)
        row["params"] = int(model.count_params())
        row["trainable_params"] = int(count_weights(model.trainable_weights))
        row["non_trainable_params"] = int(count_weights(model.non_trainable_weights))
        row["input_names"] = ";".join([getattr(x, "name", "") for x in model.inputs])
        row["input_shapes"] = ";".join([safe_shape(x) for x in model.inputs])
        row["output_names"] = ";".join([getattr(x, "name", "") for x in model.outputs])
        row["output_shapes"] = ";".join([safe_shape(x) for x in model.outputs])
    except Exception as exc:
        row["error"] = repr(exc)

    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--activity_model",
        default="models/clean_p4_final/stage2_activity_imu_single_seed42_cleanp4/best_model.h5",
    )
    p.add_argument(
        "--surface_model",
        default="models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4/best_model.h5",
    )
    p.add_argument(
        "--composite_model",
        default="models/clean_p4_final/unified_composite_cleanp4/unified_composite_cleanp4.keras",
    )
    p.add_argument(
        "--evaluation_summary",
        default="reports/clean_p4_final/unified_composite_cleanp4/evaluation_summary.json",
    )
    p.add_argument(
        "--out_dir",
        default="reports/clean_p4_final/model_complexity",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluation_summary = read_json(Path(args.evaluation_summary))

    rows = [
        describe_model(
            "clean_p4_activity_imu",
            Path(args.activity_model),
            "activity",
            ["imu"],
            extract_branch_metrics(evaluation_summary, "activity"),
        ),
        describe_model(
            "clean_p4_surface_image_audio",
            Path(args.surface_model),
            "surface",
            ["image", "audio"],
            extract_branch_metrics(evaluation_summary, "surface"),
        ),
        describe_model(
            "clean_p4_composite_wrapper",
            Path(args.composite_model),
            "engineering_wrapper",
            ["imu", "image", "audio"],
            {},
        ),
    ]

    csv_path = out_dir / "model_complexity_cleanp4.csv"
    json_path = out_dir / "model_complexity_cleanp4.json"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[DONE] Saved:")
    print(csv_path)
    print(json_path)
    print()
    print("[SUMMARY]")
    for r in rows:
        print(
            f"{r['name']}: exists={r['exists']}, "
            f"params={r['params']}, size_mb={r['model_size_mb']}, "
            f"accuracy={r['accuracy']}, macro_f1={r['macro_f1']}, "
            f"error={str(r['error'])[:160]}"
        )


if __name__ == "__main__":
    main()
