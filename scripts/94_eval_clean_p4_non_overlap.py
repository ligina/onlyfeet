#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate final Clean-P4 models on non-overlapping P4 test windows.

This is a supplementary evaluation script. It does not retrain models, modify
data, or change the final Clean-P4 model artifacts.
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
ACTIVITY_MODEL_DIR = ROOT / "models/clean_p4_final/models/clean_p4_final/stage2_activity_imu_single_seed42_cleanp4"
SURFACE_MODEL_DIR = ROOT / "models/clean_p4_final/models/clean_p4_final/stage2_surface_image_audio_concat_seed42_cleanp4"
OUT_DIR = ROOT / "reports/clean_p4_final/non_overlap"


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


def to_model_x(x_by_modality: dict, modalities: list):
    if len(modalities) == 1:
        return x_by_modality[modalities[0]]
    return [x_by_modality[m] for m in modalities]


def normalized_eval_inputs(config: dict, valid_test: np.ndarray):
    train_npz = np.load(ROOT / config["train_npz"], allow_pickle=True)
    test_npz = np.load(ROOT / config["test_npz"], allow_pickle=True)
    card = base.load_card(ROOT / config["card"])
    modalities = list(config["modalities_parsed"])
    y_train_all, valid_train, class_names = base.get_labels(train_npz, card, config["task"])
    y_train_valid = y_train_all[valid_train]
    split = clean_p4.make_internal_validation_split(
        train_npz=train_npz,
        valid_mask=valid_train,
        y=y_train_valid,
        ratio=float(config.get("validation_split_ratio", 0.15)),
        seed=int(config.get("seed", 42)),
        n_classes=len(class_names),
    )
    train_idx = split["train_idx"]

    x_test = {}
    for modality in modalities:
        x_train_all = base.load_x(train_npz, modality, imu_magnitude=not bool(config.get("no_imu_magnitude", False)))[valid_train]
        x_test_raw = base.load_x(test_npz, modality, imu_magnitude=not bool(config.get("no_imu_magnitude", False)))[valid_test]
        _, stats = clean_p4.normalize_from_train(x_train_all[train_idx])
        x_test[modality] = clean_p4.apply_normalization(x_test_raw, stats)
    return to_model_x(x_test, modalities), class_names


def select_nonoverlap_indices(folders, start_ms, min_gap_ms: float):
    rows = pd.DataFrame({
        "idx": np.arange(len(folders), dtype=int),
        "folder": np.asarray(folders).astype(str),
        "start_ms": np.asarray(start_ms, dtype=float),
    })
    selected = []
    for _, group in rows.groupby("folder", sort=False):
        last_start = None
        for _, row in group.sort_values("start_ms").iterrows():
            start = float(row["start_ms"])
            if last_start is None or start >= last_start + float(min_gap_ms):
                selected.append(int(row["idx"]))
                last_start = start
    return np.array(sorted(selected), dtype=int)


def write_metrics(out_dir: Path, prefix: str, model, x_eval, y_true, class_names, batch: int, extra: dict, metadata: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_prob = model.predict(x_eval, batch_size=batch, verbose=1)
    y_pred = np.argmax(pred_prob, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_eval": int(len(y_true)),
        "classes": class_names,
        "model_params": int(model.count_params()),
    }
    metrics.update(extra)
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    (out_dir / f"{prefix}_classification_report.txt").write_text(report_txt, encoding="utf-8")
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True)
    (out_dir / f"{prefix}_classification_report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    with (out_dir / f"{prefix}_per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for values in zip(class_names, precision, recall, f1, support):
            writer.writerow([values[0], float(values[1]), float(values[2]), float(values[3]), int(values[4])])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.save(out_dir / f"{prefix}_confusion_matrix.npy", cm)
    np.savetxt(out_dir / f"{prefix}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    np.savez_compressed(out_dir / f"{prefix}_predictions.npz", y_true=y_true, y_pred=y_pred, pred_prob=pred_prob)
    pred_rows = []
    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        row = {
            "idx": i,
            "original_index": int(metadata["original_index"][i]),
            "folder": str(metadata["folder"][i]),
            "start_ms": float(metadata["start_ms"][i]),
            "y_true": int(yt),
            "y_pred": int(yp),
            "y_true_name": class_names[int(yt)],
            "y_pred_name": class_names[int(yp)],
            "correct": int(yt == yp),
        }
        pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(out_dir / f"{prefix}_predictions.csv", index=False)
    return metrics


def evaluate_task(name: str, model_dir: Path, out_root: Path, min_gap_ms: float, batch: int):
    config = json.loads((model_dir / "train_config.json").read_text(encoding="utf-8"))
    test_npz = np.load(ROOT / config["test_npz"], allow_pickle=True)
    card = base.load_card(ROOT / config["card"])
    y_all, valid, class_names = base.get_labels(test_npz, card, config["task"])
    if "folder" not in test_npz.files or "start_ms" not in test_npz.files:
        raise KeyError(f"{config['test_npz']} must contain folder and start_ms for non-overlap evaluation")

    y_valid = y_all[valid]
    folders = np.asarray(test_npz["folder"], dtype=object)[valid].astype(str)
    start_ms = np.asarray(test_npz["start_ms"], dtype=float)[valid]
    selected = select_nonoverlap_indices(folders, start_ms, min_gap_ms)
    x_all, class_names_norm = normalized_eval_inputs(config, valid)
    if class_names != class_names_norm:
        print(f"[WARN] Class names differ for {name}; using train config class names.")
        class_names = class_names_norm
    x_selected = x_all[selected] if isinstance(x_all, np.ndarray) else [x[selected] for x in x_all]

    model = load_model_legacy(model_dir / "best_model.h5")
    task_out = out_root / name
    metrics = write_metrics(
        out_dir=task_out,
        prefix="non_overlap",
        model=model,
        x_eval=x_selected,
        y_true=y_valid[selected],
        class_names=class_names,
        batch=batch,
        extra={
            "task": config["task"],
            "modalities": config["modalities_parsed"],
            "min_gap_ms": float(min_gap_ms),
            "original_n_eval": int(len(y_valid)),
            "non_overlap_n_eval": int(len(selected)),
            "n_folders": int(len(set(folders[selected].tolist()))),
            "model_path": str(model_dir / "best_model.h5"),
            "test_npz": config["test_npz"],
            "clean_p4_scope": "P4 final test only; surface task is walking-only.",
        },
        metadata={
            "original_index": np.where(valid)[0][selected],
            "folder": folders[selected],
            "start_ms": start_ms[selected],
        },
    )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activity_model_dir", default=str(ACTIVITY_MODEL_DIR))
    parser.add_argument("--surface_model_dir", default=str(SURFACE_MODEL_DIR))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--min_gap_ms", type=float, default=2000.0)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "note": "Supplementary non-overlap evaluation. No retraining; not a jointly trained unified model.",
        "min_gap_ms": float(args.min_gap_ms),
        "activity": evaluate_task("activity", Path(args.activity_model_dir), out_root, args.min_gap_ms, args.batch),
        "surface": evaluate_task("surface", Path(args.surface_model_dir), out_root, args.min_gap_ms, args.batch),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame([summary["activity"], summary["surface"]]).to_csv(out_root / "summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
