#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect Clean-P4 model complexity metadata.

The composite model, when present, is an engineering wrapper around two
task-specific models; it is not a jointly trained unified model.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_MODEL = ROOT / "models/clean_p4_final/models/clean_p4_final/stage2_activity_imu_single_seed42_cleanp4/best_model.h5"
SURFACE_MODEL = ROOT / "models/clean_p4_final/models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4/best_model.h5"
COMPOSITE_MODEL = ROOT / "models/clean_p4_final/unified_composite_cleanp4/unified_composite_cleanp4.keras"
EVAL_SUMMARY = ROOT / "reports/clean_p4_final/unified_composite_cleanp4/evaluation_summary.json"
OUT_DIR = ROOT / "reports/clean_p4_final/model_complexity"


def load_model_legacy(path: Path):
    from tensorflow import keras

    custom_objects = {}
    if hasattr(keras.layers, "GRU"):
        custom_objects["LegacyGRU"] = keras.layers.GRU
    return keras.models.load_model(path, compile=False, custom_objects=custom_objects)


def tensor_shape(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [None if dim is None else int(dim) for dim in shape]


def tensor_name(value):
    name = getattr(value, "name", "")
    return str(name).split(":")[0] if name else None


def count_weights(weights):
    return int(sum(int(w.shape.num_elements()) for w in weights))


def load_existing_eval_metrics(path: Path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for metric_name in ["accuracy", "macro_f1", "weighted_f1", "n_eval"]:
                if metric_name in value:
                    flat[f"{key}_{metric_name}"] = value[metric_name]
        elif key in {"accuracy", "macro_f1", "weighted_f1", "n_eval"}:
            flat[key] = value
    return flat


def describe_model(name: str, path: Path, task: str, modalities: list, extra_metrics: dict):
    if not path.exists():
        return {
            "model_name": name,
            "model_path": str(path),
            "exists": False,
            "task": task,
            "modalities": ",".join(modalities),
        }
    model = load_model_legacy(path)
    row = {
        "model_name": name,
        "model_path": str(path),
        "exists": True,
        "task": task,
        "modalities": ",".join(modalities),
        "params": int(model.count_params()),
        "trainable_params": count_weights(model.trainable_weights),
        "non_trainable_params": count_weights(model.non_trainable_weights),
        "model_size_mb": float(path.stat().st_size / 1024 / 1024),
        "input_names": ",".join([tensor_name(t) or "" for t in model.inputs]),
        "input_shapes": json.dumps([tensor_shape(t) for t in model.inputs]),
        "output_names": ",".join([tensor_name(t) or "" for t in model.outputs]),
        "output_shapes": json.dumps([tensor_shape(t) for t in model.outputs]),
    }
    row.update(extra_metrics)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity_model", default=str(ACTIVITY_MODEL))
    parser.add_argument("--surface_model", default=str(SURFACE_MODEL))
    parser.add_argument("--composite_model", default=str(COMPOSITE_MODEL))
    parser.add_argument("--evaluation_summary", default=str(EVAL_SUMMARY))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    args = parser.parse_args()

    eval_metrics = load_existing_eval_metrics(Path(args.evaluation_summary))
    rows = [
        describe_model("clean_p4_activity_imu", Path(args.activity_model), "activity", ["imu"], {}),
        describe_model("clean_p4_surface_image_audio", Path(args.surface_model), "surface_walking_only", ["image", "audio"], {}),
        describe_model(
            "unified_composite_cleanp4",
            Path(args.composite_model),
            "activity_and_surface_walking_only",
            ["imu", "image", "audio"],
            eval_metrics,
        ),
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": "The composite model is an engineering wrapper, not a jointly trained unified model.",
        "models": rows,
    }
    (out_dir / "model_complexity_cleanp4.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "model_complexity_cleanp4.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
