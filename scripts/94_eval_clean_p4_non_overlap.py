#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate final Clean-P4 models on non-overlapping P4 test windows.

This is a supplementary diagnostic. It does not retrain models, modify data,
or change the final Clean-P4 artifacts.

Selection rule:
For each recording folder, sort windows by start_ms and keep windows whose
start time is at least --min_gap_ms after the last kept window. With 2-second
windows, --min_gap_ms 2000 gives a non-overlapping-window approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ACTIVITY_MODEL_DIR = ROOT / "models" / "clean_p4_final" / "stage2_activity_imu_single_seed42_cleanp4"
DEFAULT_SURFACE_MODEL_DIR = ROOT / "models" / "clean_p4_final" / "stage2_surface_image_audio_concat_seed42_cleanp4"
DEFAULT_OUT_DIR = ROOT / "reports" / "clean_p4_final" / "non_overlap"

NPZ_KEYS = {
    "imu": "imu_win",
    "image": "img_win",
    "audio": "audio_win",
    "tof": "tof_win",
    "mag": "mag_win",
}


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
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return ROOT / p


def load_model_legacy(path: Path):
    return keras.models.load_model(
        path,
        compile=False,
        custom_objects=LEGACY_CUSTOM_OBJECTS,
        safe_mode=False,
    )


def ensure_audio_channel(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.ndim == 3:
        x = x[..., np.newaxis]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def ensure_image_float(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if np.nanmax(x) > 2.0:
        x = x / 255.0
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def add_imu_magnitudes(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 3:
        return x
    if x.shape[-1] >= 8:
        return x
    if x.shape[-1] == 7:
        gyro = x[..., 3:6]
        gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=-1, keepdims=True))
        return np.concatenate([x, gyro_mag], axis=-1).astype(np.float32)
    if x.shape[-1] == 6:
        acc = x[..., 0:3]
        gyro = x[..., 3:6]
        acc_mag = np.sqrt(np.sum(acc ** 2, axis=-1, keepdims=True))
        gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=-1, keepdims=True))
        return np.concatenate([x, acc_mag, gyro_mag], axis=-1).astype(np.float32)
    return x


def load_x(npz: np.lib.npyio.NpzFile, modality: str, imu_magnitude: bool = True) -> np.ndarray:
    key = NPZ_KEYS[modality]
    if key not in npz.files:
        raise KeyError(f"Missing {key} for modality={modality}. Available: {npz.files}")

    x = npz[key]
    if modality == "audio":
        return ensure_audio_channel(x)
    if modality == "image":
        return ensure_image_float(x)
    if modality == "imu" and imu_magnitude:
        return add_imu_magnitudes(x)
    return np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def class_names_from_config_or_card(config: dict[str, Any], card_path: Path, task: str) -> list[str]:
    if "class_names" in config and config["class_names"]:
        return list(config["class_names"])

    card = read_json(card_path)
    if task == "activity":
        for key in ["activity_classes", "act_classes", "classes_activity"]:
            if key in card:
                return list(card[key])
    if task == "surface":
        for key in ["env_classes", "surface_classes", "environment_classes"]:
            if key in card:
                return list(card[key])

    raise KeyError(f"Cannot infer class names for task={task} from config or {card_path}")


def get_labels(npz: np.lib.npyio.NpzFile, class_names: list[str], task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "activity":
        y = npz["y_act"].astype(np.int64)
    elif task == "surface":
        y = npz["y_env"].astype(np.int64)
    else:
        raise ValueError(f"Unsupported task: {task}")

    valid = (y >= 0) & (y < len(class_names))
    return y, valid


def parse_modalities(config: dict[str, Any]) -> list[str]:
    if "modalities_parsed" in config:
        return [str(x).lower() for x in config["modalities_parsed"]]
    if "modalities" in config:
        v = config["modalities"]
        if isinstance(v, list):
            return [str(x).lower() for x in v]
        return [x.strip().lower() for x in str(v).split(",") if x.strip()]
    raise KeyError("Cannot infer modalities from train_config.json")


def choose_folder_split(
    y: np.ndarray,
    folders: np.ndarray,
    ratio: float,
    seed: int,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(np.unique(folders)) < 2:
        return None

    splitter = GroupShuffleSplit(n_splits=200, test_size=ratio, random_state=seed)
    indices = np.arange(len(y))
    best = None
    best_score = None

    for train_idx, val_idx in splitter.split(indices, y, groups=folders):
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        train_classes = set(int(v) for v in np.unique(y[train_idx]))
        val_classes = set(int(v) for v in np.unique(y[val_idx]))
        missing = (n_classes - len(train_classes)) + (n_classes - len(val_classes))
        ratio_error = abs((len(val_idx) / len(y)) - ratio)
        score = (missing, ratio_error, len(np.unique(folders[val_idx])))
        if best_score is None or score < best_score:
            best_score = score
            best = (train_idx, val_idx)
            if missing == 0 and ratio_error <= 0.01:
                break

    return best


def choose_sample_split(y: np.ndarray, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    class_counts = np.bincount(y)
    if np.any(class_counts[class_counts > 0] < 2):
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(y) * ratio)))
        return np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val])

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, y))
    return np.sort(train_idx), np.sort(val_idx)


def internal_train_indices(
    train_npz: np.lib.npyio.NpzFile,
    valid_mask: np.ndarray,
    y_valid: np.ndarray,
    config: dict[str, Any],
    n_classes: int,
) -> np.ndarray:
    ratio = float(config.get("validation_split_ratio", 0.15))
    seed = int(config.get("seed", 42))

    if "folder" in train_npz.files:
        folders = np.asarray(train_npz["folder"], dtype=object)[valid_mask].astype(str)
        split = choose_folder_split(y_valid, folders, ratio, seed, n_classes)
        if split is not None:
            train_idx, _ = split
            return np.sort(train_idx)

    train_idx, _ = choose_sample_split(y_valid, ratio, seed)
    return np.sort(train_idx)


def normalization_stats(
    train_npz_path: Path,
    config: dict[str, Any],
    task: str,
    class_names: list[str],
    modalities: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    train_npz = np.load(train_npz_path, allow_pickle=True)
    y_all, valid_mask = get_labels(train_npz, class_names, task)
    y_valid = y_all[valid_mask]
    train_idx = internal_train_indices(train_npz, valid_mask, y_valid, config, len(class_names))

    stats: dict[str, dict[str, np.ndarray]] = {}
    imu_magnitude = not bool(config.get("no_imu_magnitude", False))

    for modality in modalities:
        raw_valid = load_x(train_npz, modality, imu_magnitude=imu_magnitude)[valid_mask]
        x_train = raw_valid[train_idx]
        axes = tuple(range(x_train.ndim - 1))
        stats[modality] = {
            "mean": x_train.mean(axis=axes, keepdims=True),
            "std": x_train.std(axis=axes, keepdims=True) + 1e-6,
        }

    return stats


def apply_normalization(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    x = (x - stats["mean"]) / stats["std"]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def non_overlap_indices(
    test_npz: np.lib.npyio.NpzFile,
    valid_mask: np.ndarray,
    min_gap_ms: float,
) -> np.ndarray:
    if "folder" not in test_npz.files or "start_ms" not in test_npz.files:
        raise KeyError("Non-overlap evaluation requires both 'folder' and 'start_ms' arrays in the test NPZ.")

    valid_indices = np.where(valid_mask)[0]
    folders = np.asarray(test_npz["folder"], dtype=object)[valid_indices].astype(str)
    start_ms = np.asarray(test_npz["start_ms"], dtype=np.float64)[valid_indices]

    kept_local: list[int] = []
    for folder in sorted(np.unique(folders)):
        local = np.where(folders == folder)[0]
        local = local[np.argsort(start_ms[local])]

        last_kept = None
        for idx in local:
            t = float(start_ms[idx])
            if last_kept is None or (t - last_kept) >= float(min_gap_ms):
                kept_local.append(int(idx))
                last_kept = t

    return np.array(sorted(kept_local), dtype=np.int64)


def infer_input_modality(input_name: str, fallback: str | None = None) -> str:
    n = input_name.lower()
    if "imu" in n:
        return "imu"
    if "image" in n or "img" in n or "rgb" in n:
        return "image"
    if "audio" in n or "aud" in n:
        return "audio"
    if "tof" in n:
        return "tof"
    if "mag" in n:
        return "mag"
    if fallback is not None:
        return fallback
    raise ValueError(f"Cannot infer modality from model input name: {input_name}")


def build_model_input(model, x_by_modality: dict[str, np.ndarray]):
    if len(model.inputs) == 1:
        mod = infer_input_modality(model.inputs[0].name, fallback=next(iter(x_by_modality.keys())))
        return x_by_modality[mod]

    xs = []
    for inp in model.inputs:
        mod = infer_input_modality(inp.name)
        xs.append(x_by_modality[mod])
    return xs


def normalize_probabilities(prob: Any) -> np.ndarray:
    if isinstance(prob, (list, tuple)):
        if len(prob) != 1:
            raise ValueError(f"Expected single-output model, got {len(prob)} outputs")
        prob = prob[0]
    if isinstance(prob, dict):
        if len(prob) != 1:
            raise ValueError(f"Expected single-output dict, got keys={list(prob.keys())}")
        prob = next(iter(prob.values()))

    prob = np.asarray(prob)
    if prob.ndim == 3 and prob.shape[0] == 1:
        prob = prob[0]
    if prob.ndim == 3 and prob.shape[1] == 1:
        prob = prob[:, 0, :]
    if prob.ndim != 2:
        raise ValueError(f"Expected probabilities with shape (N, C), got {prob.shape}")
    return prob


def write_task_outputs(
    out_dir: Path,
    prefix: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_prob: np.ndarray,
    class_names: list[str],
    extra_metrics: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        **extra_metrics,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_eval": int(len(y_true)),
        "classes": class_names,
    }

    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    (out_dir / f"{prefix}_classification_report.txt").write_text(report_txt, encoding="utf-8")

    report_json = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True)
    (out_dir / f"{prefix}_classification_report.json").write_text(json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.save(out_dir / f"{prefix}_confusion_matrix.npy", cm)
    np.savetxt(out_dir / f"{prefix}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    with (out_dir / f"{prefix}_per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for name, pp, rr, ff, ss in zip(class_names, precision, recall, f1, support):
            writer.writerow([name, float(pp), float(rr), float(ff), int(ss)])

    np.savez_compressed(
        out_dir / f"{prefix}_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        pred_prob=pred_prob,
    )
    with (out_dir / f"{prefix}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "y_true", "y_pred", "y_true_name", "y_pred_name", "correct"])
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            writer.writerow([i, int(yt), int(yp), class_names[int(yt)], class_names[int(yp)], int(yt == yp)])

    return metrics


def evaluate_task(task: str, model_dir: Path, out_root: Path, min_gap_ms: float, batch: int) -> dict[str, Any]:
    config = read_json(model_dir / "train_config.json")
    model_path = model_dir / "best_model.h5"
    train_npz_path = resolve_path(config["train_npz"])
    test_npz_path = resolve_path(config["test_npz"])
    card_path = resolve_path(config["card"])

    task_from_config = str(config.get("task", task)).lower()
    if task_from_config in ["env", "environment"]:
        task_from_config = "surface"

    class_names = class_names_from_config_or_card(config, card_path, task_from_config)
    modalities = parse_modalities(config)

    train_stats = normalization_stats(
        train_npz_path=train_npz_path,
        config=config,
        task=task_from_config,
        class_names=class_names,
        modalities=modalities,
    )

    test_npz = np.load(test_npz_path, allow_pickle=True)
    y_all, valid_mask = get_labels(test_npz, class_names, task_from_config)
    y_valid = y_all[valid_mask]

    kept_local = non_overlap_indices(test_npz, valid_mask, min_gap_ms=min_gap_ms)
    y_true = y_valid[kept_local]

    imu_magnitude = not bool(config.get("no_imu_magnitude", False))
    x_by_modality: dict[str, np.ndarray] = {}
    for modality in modalities:
        raw_valid = load_x(test_npz, modality, imu_magnitude=imu_magnitude)[valid_mask]
        raw_kept = raw_valid[kept_local]
        x_by_modality[modality] = apply_normalization(raw_kept, train_stats[modality])

    model = load_model_legacy(model_path)
    pred_prob = normalize_probabilities(
        model.predict(build_model_input(model, x_by_modality), batch_size=int(batch), verbose=1)
    )
    y_pred = np.argmax(pred_prob, axis=1)

    out_dir = out_root / task_from_config
    metrics = write_task_outputs(
        out_dir=out_dir,
        prefix=task_from_config,
        y_true=y_true,
        y_pred=y_pred,
        pred_prob=pred_prob,
        class_names=class_names,
        extra_metrics={
            "task": task_from_config,
            "modalities": modalities,
            "model_path": str(model_path),
            "train_npz": str(train_npz_path),
            "test_npz": str(test_npz_path),
            "original_n_eval": int(np.sum(valid_mask)),
            "non_overlap_n_eval": int(len(y_true)),
            "min_gap_ms": float(min_gap_ms),
            "model_params": int(model.count_params()),
        },
    )
    return metrics


def write_summary(out_root: Path, summary: dict[str, Any]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for key in ["activity", "surface"]:
        if key in summary:
            rows.append(summary[key])

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with (out_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--activity_model_dir", default=str(DEFAULT_ACTIVITY_MODEL_DIR))
    p.add_argument("--surface_model_dir", default=str(DEFAULT_SURFACE_MODEL_DIR))
    p.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--min_gap_ms", type=float, default=2000.0)
    p.add_argument("--batch", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "note": "Supplementary non-overlapping-window evaluation. No retraining; no data modification.",
        "min_gap_ms": float(args.min_gap_ms),
        "activity": evaluate_task("activity", Path(args.activity_model_dir), out_root, args.min_gap_ms, args.batch),
        "surface": evaluate_task("surface", Path(args.surface_model_dir), out_root, args.min_gap_ms, args.batch),
    }

    write_summary(out_root, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
