#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a Clean-P4 composite Keras wrapper model.

This script packages the two Clean-P4 task-specific final models behind one
Keras Functional model with separate inputs and two independent outputs:

* act_output: activity recognition from IMU-only input.
* surface_output: walking-only surface recognition from image+audio inputs.

Important scope note: this is an engineering wrapper around two already-trained
specialist models. It is not a jointly trained unified model, does not change
model weights or results, and does not broaden the walking-only scope of the
surface recognizer.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_P4_ROOT = REPO_ROOT / "models" / "clean_p4_final"
ACTIVITY_DIR = CLEAN_P4_ROOT / "stage2_activity_imu_single_seed42_cleanp4"
SURFACE_DIR = CLEAN_P4_ROOT / "stage2_surface_image_audio_concat_seed42_cleanp4"
DEFAULT_MODEL_OUT_DIR = CLEAN_P4_ROOT / "unified_composite_cleanp4"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "clean_p4_final" / "unified_composite_cleanp4"

NPZ_KEYS = {
    "imu": "imu_win",
    "image": "img_win",
    "audio": "audio_win",
}


class EvaluationSkipped(RuntimeError):
    """Raised when a requested evaluation cannot be run without inventing metrics."""


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ensure_audio_channel(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.ndim == 3:
        x = x[..., np.newaxis]
    return x


def ensure_image_float(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if np.nanmax(x) > 2.0:
        x = x / 255.0
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def add_imu_magnitudes(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim != 3 or x.shape[-1] < 6 or x.shape[-1] >= 8:
        return x
    acc = x[..., 0:3]
    gyro = x[..., 3:6]
    acc_mag = np.sqrt(np.sum(acc**2, axis=-1, keepdims=True))
    gyro_mag = np.sqrt(np.sum(gyro**2, axis=-1, keepdims=True))
    return np.concatenate([x, acc_mag, gyro_mag], axis=-1).astype(np.float32)


def load_x(npz: np.lib.npyio.NpzFile, modality: str, imu_magnitude: bool = True) -> np.ndarray:
    key = NPZ_KEYS[modality]
    if key not in npz.files:
        raise KeyError(f"Missing {key} for modality {modality}. Available keys: {npz.files}")
    x = npz[key]
    if modality == "audio":
        return ensure_audio_channel(x)
    if modality == "image":
        return ensure_image_float(x)
    if modality == "imu" and imu_magnitude:
        return add_imu_magnitudes(x)
    return np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def get_labels(npz: np.lib.npyio.NpzFile, class_names: list[str], task: str) -> tuple[np.ndarray, np.ndarray]:
    if task == "activity":
        y = npz["y_act"].astype(np.int64)
    elif task == "surface":
        y = npz["y_env"].astype(np.int64)
    else:
        raise ValueError(f"Unsupported task: {task}")
    valid = (y >= 0) & (y < len(class_names))
    return y, valid


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
    best = None
    best_score = None
    indices = np.arange(len(y))

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
) -> np.ndarray:
    ratio = float(config.get("validation_split_ratio", 0.15))
    seed = int(config.get("seed", 42))
    n_classes = len(config["class_names"])

    if "folder" in train_npz.files:
        folders = np.asarray(train_npz["folder"], dtype=object)[valid_mask].astype(str)
        split = choose_folder_split(y_valid, folders, ratio, seed, n_classes)
        if split is not None:
            train_idx, _ = split
            return np.sort(train_idx)

    train_idx, _ = choose_sample_split(y_valid, ratio, seed)
    return train_idx


def normalization_from_internal_train(
    train_npz_path: Path,
    config: dict[str, Any],
    task: str,
    modalities: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    if not train_npz_path.exists():
        raise EvaluationSkipped(
            f"Training NPZ is required to recreate Clean-P4 normalization, but it was not found: {train_npz_path}"
        )

    train_npz = np.load(train_npz_path, allow_pickle=True)
    y_all, valid_mask = get_labels(train_npz, config["class_names"], task)
    y_valid = y_all[valid_mask]
    train_idx = internal_train_indices(train_npz, valid_mask, y_valid, config)

    stats = {}
    for modality in modalities:
        raw_valid = load_x(train_npz, modality, imu_magnitude=not bool(config.get("no_imu_magnitude", False)))[valid_mask]
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


def load_normalized_test_inputs(
    test_npz_path: Path,
    train_npz_path: Path,
    config: dict[str, Any],
    task: str,
    modalities: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not test_npz_path.exists():
        raise EvaluationSkipped(f"Test NPZ was requested but not found: {test_npz_path}")

    stats = normalization_from_internal_train(train_npz_path, config, task, modalities)
    test_npz = np.load(test_npz_path, allow_pickle=True)
    y_all, valid_mask = get_labels(test_npz, config["class_names"], task)
    y_true = y_all[valid_mask]

    x_by_modality = {}
    for modality in modalities:
        raw = load_x(test_npz, modality, imu_magnitude=not bool(config.get("no_imu_magnitude", False)))[valid_mask]
        x_by_modality[modality] = apply_normalization(raw, stats[modality])
    return x_by_modality, y_true


def zeros_for(inputs: dict[str, tuple[int, ...]], n_samples: int) -> dict[str, np.ndarray]:
    return {name: np.zeros((n_samples, *shape), dtype=np.float32) for name, shape in inputs.items()}


def metrics_from_probabilities(
    pred_prob: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    model_params: int,
    out_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = np.argmax(pred_prob, axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_eval": int(len(y_true)),
        "classes": class_names,
        "model_params": int(model_params),
    }
    write_json(out_dir / f"{prefix}_metrics.json", metrics)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True)
    write_json(out_dir / f"{prefix}_classification_report.json", report_dict)
    (out_dir / f"{prefix}_classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0),
        encoding="utf-8",
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    with (out_dir / f"{prefix}_per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for name, p, r, f, s in zip(class_names, precision, recall, f1, support):
            writer.writerow([name, float(p), float(r), float(f), int(s)])

    np.savez_compressed(out_dir / f"{prefix}_predictions.npz", y_true=y_true, y_pred=y_pred, pred_prob=pred_prob)
    with (out_dir / f"{prefix}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "y_true", "y_pred", "y_true_name", "y_pred_name", "correct"])
        for idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
            writer.writerow([idx, int(yt), int(yp), class_names[int(yt)], class_names[int(yp)], int(yt == yp)])
    return metrics


def build_composite_model(activity_model_path: Path, surface_model_path: Path):
    from tensorflow import keras

    activity_model = keras.models.load_model(activity_model_path, compile=False)
    surface_model = keras.models.load_model(surface_model_path, compile=False)
    activity_model.trainable = False
    surface_model.trainable = False

    act_input = keras.Input(shape=tuple(activity_model.inputs[0].shape[1:]), name="activity_imu_input")
    surface_image_input = keras.Input(shape=tuple(surface_model.inputs[0].shape[1:]), name="surface_image_input")
    surface_audio_input = keras.Input(shape=tuple(surface_model.inputs[1].shape[1:]), name="surface_audio_input")

    act_pred = activity_model(act_input)
    surface_pred = surface_model([surface_image_input, surface_audio_input])
    act_output = keras.layers.Activation("linear", name="act_output")(act_pred)
    surface_output = keras.layers.Activation("linear", name="surface_output")(surface_pred)

    composite = keras.Model(
        inputs=[act_input, surface_image_input, surface_audio_input],
        outputs=[act_output, surface_output],
        name="unified_composite_cleanp4_engineering_wrapper",
    )
    return composite, activity_model, surface_model


def load_evaluation_dependencies() -> None:
    global np, accuracy_score, classification_report, f1_score, precision_recall_fscore_support
    global GroupShuffleSplit, StratifiedShuffleSplit

    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_recall_fscore_support
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


def evaluate_composite(
    composite,
    activity_model,
    surface_model,
    args: argparse.Namespace,
    activity_config: dict[str, Any],
    surface_config: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "scope_note": "Evaluation uses the composite wrapper outputs independently; the source task models remain separately trained.",
        "surface_scope": "walking-only surface recognition",
        "activity": {"status": "not_requested"},
        "surface": {"status": "not_requested"},
    }

    input_shapes = {
        "activity_imu_input": tuple(composite.inputs[0].shape[1:]),
        "surface_image_input": tuple(composite.inputs[1].shape[1:]),
        "surface_audio_input": tuple(composite.inputs[2].shape[1:]),
    }

    if args.activity_test_npz or args.surface_test_npz:
        load_evaluation_dependencies()

    if args.activity_test_npz:
        try:
            act_x, y_act = load_normalized_test_inputs(
                Path(args.activity_test_npz),
                Path(args.activity_train_npz),
                activity_config,
                "activity",
                ["imu"],
            )
            n = len(y_act)
            inputs = zeros_for(input_shapes, n)
            inputs["activity_imu_input"] = act_x["imu"]
            pred_prob = composite.predict(inputs, batch_size=args.batch, verbose=args.verbose)[0]
            metrics = metrics_from_probabilities(
                pred_prob,
                y_act,
                activity_config["class_names"],
                activity_model.count_params(),
                Path(args.report_dir),
                "activity_eval",
            )
            metrics.update({"task": "activity", "modalities": ["imu"], "source_test_npz": args.activity_test_npz})
            summary["activity"] = {"status": "completed", "metrics": metrics}
        except EvaluationSkipped as exc:
            summary["activity"] = {"status": "skipped", "reason": str(exc)}

    if args.surface_test_npz:
        try:
            surface_x, y_surface = load_normalized_test_inputs(
                Path(args.surface_test_npz),
                Path(args.surface_train_npz),
                surface_config,
                "surface",
                ["image", "audio"],
            )
            n = len(y_surface)
            inputs = zeros_for(input_shapes, n)
            inputs["surface_image_input"] = surface_x["image"]
            inputs["surface_audio_input"] = surface_x["audio"]
            pred_prob = composite.predict(inputs, batch_size=args.batch, verbose=args.verbose)[1]
            metrics = metrics_from_probabilities(
                pred_prob,
                y_surface,
                surface_config["class_names"],
                surface_model.count_params(),
                Path(args.report_dir),
                "surface_eval",
            )
            metrics.update({
                "task": "surface",
                "modalities": ["image", "audio"],
                "source_test_npz": args.surface_test_npz,
                "scope": "walking-only surface samples",
            })
            summary["surface"] = {"status": "completed", "metrics": metrics}
        except EvaluationSkipped as exc:
            summary["surface"] = {"status": "skipped", "reason": str(exc)}

    return summary


def default_path_from_config(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    return str(REPO_ROOT / value) if value and not Path(value).is_absolute() else str(value or "")


def parse_args() -> argparse.Namespace:
    activity_config = read_json(ACTIVITY_DIR / "train_config.json")
    surface_config = read_json(SURFACE_DIR / "train_config.json")

    parser = argparse.ArgumentParser(
        description="Create a Clean-P4 composite Keras wrapper around the final activity and walking-only surface models."
    )
    parser.add_argument("--activity_model", default=str(ACTIVITY_DIR / "final_model.h5"))
    parser.add_argument("--surface_model", default=str(SURFACE_DIR / "final_model.h5"))
    parser.add_argument("--model_out_dir", default=str(DEFAULT_MODEL_OUT_DIR))
    parser.add_argument("--report_dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--activity_test_npz", default=None, help="Optional activity dataset_test.npz for wrapper evaluation.")
    parser.add_argument("--surface_test_npz", default=None, help="Optional walking-only surface dataset_test.npz for wrapper evaluation.")
    parser.add_argument("--activity_train_npz", default=default_path_from_config(activity_config, "train_npz"))
    parser.add_argument("--surface_train_npz", default=default_path_from_config(surface_config, "train_npz"))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def write_report_markdown(report_dir: Path, metadata: dict[str, Any], evaluation_summary: dict[str, Any]) -> None:
    lines = [
        "# Clean-P4 Unified Composite Wrapper",
        "",
        "This artifact is an engineering wrapper around two separately trained Clean-P4 final models.",
        "It is not a jointly trained unified model and does not change model weights, data, or reported task results.",
        "",
        "## Outputs",
        "",
        "- `act_output`: activity recognition from IMU-only input.",
        "- `surface_output`: walking-only surface recognition from image+audio inputs.",
        "",
        "## Saved model",
        "",
        f"- `{metadata['saved_model']}`",
        "",
        "## Evaluation summary",
        "",
        f"- Activity evaluation status: `{evaluation_summary['activity']['status']}`",
        f"- Surface evaluation status: `{evaluation_summary['surface']['status']}`",
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    activity_model_path = Path(args.activity_model)
    surface_model_path = Path(args.surface_model)
    model_out_dir = Path(args.model_out_dir)
    report_dir = Path(args.report_dir)

    activity_config = read_json(ACTIVITY_DIR / "train_config.json")
    surface_config = read_json(SURFACE_DIR / "train_config.json")

    model_out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    composite, activity_model, surface_model = build_composite_model(activity_model_path, surface_model_path)
    saved_model_path = model_out_dir / "unified_composite_cleanp4.keras"
    composite.save(saved_model_path)

    metadata = {
        "artifact_type": "Clean-P4 composite Keras engineering wrapper",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "saved_model": repo_relative(saved_model_path),
        "not_jointly_trained": True,
        "weights_changed": False,
        "data_changed": False,
        "scope_note": "This wraps two task-specific final models; it is not a jointly trained unified model.",
        "surface_scope": "walking-only surface recognition only",
        "inputs": [
            {"name": "activity_imu_input", "source_task": "activity", "modality": "imu", "shape": list(composite.inputs[0].shape[1:])},
            {"name": "surface_image_input", "source_task": "surface", "modality": "image", "shape": list(composite.inputs[1].shape[1:])},
            {"name": "surface_audio_input", "source_task": "surface", "modality": "audio", "shape": list(composite.inputs[2].shape[1:])},
        ],
        "outputs": [
            {"name": "act_output", "classes": activity_config["class_names"]},
            {"name": "surface_output", "classes": surface_config["class_names"], "scope": "walking-only"},
        ],
        "source_models": {
            "activity": {
                "path": repo_relative(activity_model_path),
                "task": "activity",
                "modalities": ["imu"],
                "class_names": activity_config["class_names"],
            },
            "surface": {
                "path": repo_relative(surface_model_path),
                "task": "surface",
                "modalities": ["image", "audio"],
                "class_names": surface_config["class_names"],
                "scope": "walking-only surface samples",
            },
        },
    }
    write_json(report_dir / "metadata.json", metadata)

    evaluation_summary = evaluate_composite(composite, activity_model, surface_model, args, activity_config, surface_config)
    write_json(report_dir / "evaluation_summary.json", evaluation_summary)
    write_report_markdown(report_dir, metadata, evaluation_summary)

    print(json.dumps({"metadata": metadata, "evaluation_summary": evaluation_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
