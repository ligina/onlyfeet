#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Clean-P4 walking-only surface model under modality ablations.

Conditions:
- normal: image and audio unchanged after train-set normalization
- no_image: normalized image input replaced with zeros_like
- no_audio: normalized audio input replaced with zeros_like

This script does not retrain models, modify data, or make deployment claims.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
SURFACE_MODEL_DIR = ROOT / "models/clean_p4_final/models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4"
OUT_DIR = ROOT / "reports/clean_p4_final/robustness_surface"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module("m14_task_model_base", ROOT / "scripts/02_train_m14_task_model.py")
clean_p4 = load_module("clean_p4_trainer", ROOT / "scripts/02_train_m14_task_model_clean_p4.py")


def load_model_legacy(path: Path):
    from tensorflow import keras

    custom_objects = {}
    if hasattr(keras.layers, "GRU"):
        custom_objects["LegacyGRU"] = keras.layers.GRU
    return keras.models.load_model(path, compile=False, custom_objects=custom_objects)


def normalized_surface_inputs(config: dict, valid_test: np.ndarray):
    train_npz = np.load(ROOT / config["train_npz"], allow_pickle=True)
    test_npz = np.load(ROOT / config["test_npz"], allow_pickle=True)
    card = base.load_card(ROOT / config["card"])
    y_train_all, valid_train, class_names = base.get_labels(train_npz, card, "surface")
    y_train_valid = y_train_all[valid_train]
    split = clean_p4.make_internal_validation_split(
        train_npz=train_npz,
        valid_mask=valid_train,
        y=y_train_valid,
        ratio=float(config.get("validation_split_ratio", 0.15)),
        seed=int(config.get("seed", 42)),
        n_classes=len(class_names),
    )
    x_test = {}
    for modality in config["modalities_parsed"]:
        x_train_all = base.load_x(train_npz, modality, imu_magnitude=not bool(config.get("no_imu_magnitude", False)))[valid_train]
        x_test_raw = base.load_x(test_npz, modality, imu_magnitude=not bool(config.get("no_imu_magnitude", False)))[valid_test]
        _, stats = clean_p4.normalize_from_train(x_train_all[split["train_idx"]])
        x_test[modality] = clean_p4.apply_normalization(x_test_raw, stats)
    return x_test, class_names


def make_condition_inputs(x_by_modality: dict, modalities: list, condition: str):
    conditioned = {key: value.copy() for key, value in x_by_modality.items()}
    if condition == "no_image":
        conditioned["image"] = np.zeros_like(conditioned["image"])
    elif condition == "no_audio":
        conditioned["audio"] = np.zeros_like(conditioned["audio"])
    elif condition != "normal":
        raise ValueError(f"Unknown condition: {condition}")
    return [conditioned[m] for m in modalities]


def write_condition(out_dir: Path, condition: str, model, x_eval, y_true, class_names, batch: int):
    cond_dir = out_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    pred_prob = model.predict(x_eval, batch_size=batch, verbose=1)
    y_pred = np.argmax(pred_prob, axis=1)
    metrics = {
        "condition": condition,
        "task": "surface",
        "surface_scope": "walking-only",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_eval": int(len(y_true)),
        "classes": class_names,
        "model_params": int(model.count_params()),
    }
    (cond_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    (cond_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True)
    (cond_dir / "classification_report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    with (cond_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for values in zip(class_names, precision, recall, f1, support):
            writer.writerow([values[0], float(values[1]), float(values[2]), float(values[3]), int(values[4])])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.save(cond_dir / "confusion_matrix.npy", cm)
    np.savetxt(cond_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    np.savez_compressed(cond_dir / "predictions.npz", y_true=y_true, y_pred=y_pred, pred_prob=pred_prob)
    rows = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        rows.append({
            "idx": i,
            "y_true": int(yt),
            "y_pred": int(yp),
            "y_true_name": class_names[int(yt)],
            "y_pred_name": class_names[int(yp)],
            "correct": int(yt == yp),
        })
    pd.DataFrame(rows).to_csv(cond_dir / "predictions.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=str(SURFACE_MODEL_DIR))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((model_dir / "train_config.json").read_text(encoding="utf-8"))
    if config["task"] != "surface" or list(config["modalities_parsed"]) != ["image", "audio"]:
        raise ValueError("Expected Clean-P4 surface image+audio model config.")

    test_npz = np.load(ROOT / config["test_npz"], allow_pickle=True)
    card = base.load_card(ROOT / config["card"])
    y_all, valid_test, class_names = base.get_labels(test_npz, card, "surface")
    x_by_modality, train_class_names = normalized_surface_inputs(config, valid_test)
    if class_names != train_class_names:
        print("[WARN] Class names differ; using train config class names.")
        class_names = train_class_names
    y_true = y_all[valid_test]
    model = load_model_legacy(model_dir / "best_model.h5")

    summary = {
        "note": "Supplementary modality robustness evaluation. No retraining; surface task is walking-only.",
        "model_path": str(model_dir / "best_model.h5"),
        "test_npz": config["test_npz"],
        "conditions": {},
    }
    for condition in ["normal", "no_image", "no_audio"]:
        x_eval = make_condition_inputs(x_by_modality, config["modalities_parsed"], condition)
        summary["conditions"][condition] = write_condition(out_dir, condition, model, x_eval, y_true, class_names, args.batch)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summary["conditions"].values()).to_csv(out_dir / "summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
