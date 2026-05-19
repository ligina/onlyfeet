#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NPZ_KEYS = {
    "imu": "imu_win",
    "tof": "tof_win",
    "mag": "mag_win",
    "audio": "audio_win",
    "image": "img_win",
}

ALL_MODALITIES = ["imu", "tof", "mag", "audio", "image"]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_audio_channel(x):
    x = x.astype(np.float32)
    if x.ndim == 3:
        x = x[..., np.newaxis]
    return x


def ensure_image_float(x):
    x = x.astype(np.float32)
    if np.nanmax(x) > 2.0:
        x = x / 255.0
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def add_imu_magnitudes(x):
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim == 3 and x.shape[-1] == 6:
        acc = x[..., :3]
        gyro = x[..., 3:6]
        acc_mag = np.sqrt(np.sum(acc ** 2, axis=-1, keepdims=True))
        gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=-1, keepdims=True))
        return np.concatenate([x, acc_mag, gyro_mag], axis=-1).astype(np.float32)
    return x


def normalize_train_eval(x_train, x_eval):
    axes = tuple(range(x_train.ndim - 1))
    mean = x_train.mean(axis=axes, keepdims=True)
    std = x_train.std(axis=axes, keepdims=True) + 1e-6
    x_eval = (x_eval - mean) / std
    return np.nan_to_num(x_eval, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def infer_modality_from_shape(expected):
    e = tuple(expected)
    if e == (50, 8) or e == (50, 6):
        return "imu"
    if e == (50, 2):
        return "tof"
    if e == (50, 3):
        return "mag"
    if e == (40, 40, 1):
        return "audio"
    if e == (64, 64, 6):
        return "image"
    return None


def expected_shape_without_batch(inp):
    shp = inp.shape
    if hasattr(shp, "as_list"):
        return tuple(shp.as_list()[1:])
    return tuple(shp[1:])


def load_modality(npz, modality, no_imu_magnitude=False):
    arr = npz[NPZ_KEYS[modality]]
    if modality == "audio":
        arr = ensure_audio_channel(arr)
    elif modality == "image":
        arr = ensure_image_float(arr)
    elif modality == "imu" and not no_imu_magnitude:
        arr = add_imu_magnitudes(arr)
    else:
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def get_labels(npz, card, task):
    if task == "activity":
        y = npz["y_act"].astype(int)
        names = card.get("activity_classes", card.get("act_classes"))
    else:
        y = npz["y_env"].astype(int)
        names = card.get("env_classes", card.get("surface_classes", card.get("environment_classes")))
    valid = y >= 0
    return y[valid], valid, list(names)


def save_cm(cm, labels, path, title, normalize=False):
    data = cm.astype(float)
    if normalize:
        data = data / np.maximum(data.sum(axis=1, keepdims=True), 1.0)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), max(5, len(labels) * 0.9)))
    im = ax.imshow(data)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fmt = ".2f" if normalize else ".0f"
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(data[i, j], fmt), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate_condition(model, x_inputs, y_true, labels, out_dir, condition, batch):
    out_dir.mkdir(parents=True, exist_ok=True)

    prob = model.predict(x_inputs, batch_size=batch, verbose=1)
    pred = np.argmax(prob, axis=1)

    metrics = {
        "condition": condition,
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "n_eval": int(len(y_true)),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(
        classification_report(y_true, pred, target_names=labels, digits=4, zero_division=0),
        encoding="utf-8",
    )

    p, r, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=list(range(len(labels))), zero_division=0
    )
    with (out_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for row in zip(labels, p, r, f1, support):
            w.writerow([row[0], float(row[1]), float(row[2]), float(row[3]), int(row[4])])

    cm = confusion_matrix(y_true, pred, labels=list(range(len(labels))))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    save_cm(cm, labels, out_dir / "confusion_matrix.png", f"{condition} confusion matrix", False)
    save_cm(cm, labels, out_dir / "confusion_matrix_normalized.png", f"{condition} normalized confusion matrix", True)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--eval_npz", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--task", required=True, choices=["activity", "surface"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--conditions", nargs="*", default=["normal"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no_imu_magnitude", action="store_true")
    args = ap.parse_args()

    model = keras.models.load_model(args.model, compile=False)
    train_npz = np.load(args.train_npz, allow_pickle=True)
    eval_npz = np.load(args.eval_npz, allow_pickle=True)
    card = load_json(args.card)

    y_true, valid, labels = get_labels(eval_npz, card, args.task)

    base_arrays = []
    input_info = []

    for inp in model.inputs:
        expected = expected_shape_without_batch(inp)
        modality = infer_modality_from_shape(expected)
        if modality is None:
            raise ValueError(f"Cannot infer modality from shape={expected}, input={inp.name}")

        x_train = load_modality(train_npz, modality, no_imu_magnitude=args.no_imu_magnitude)
        x_eval = load_modality(eval_npz, modality, no_imu_magnitude=args.no_imu_magnitude)

        x_eval = normalize_train_eval(x_train, x_eval)[valid]

        if tuple(x_eval.shape[1:]) != tuple(expected):
            raise ValueError(f"Shape mismatch for {modality}: got={x_eval.shape[1:]}, expected={expected}")

        base_arrays.append(x_eval)
        input_info.append({
            "input_name": inp.name,
            "modality": modality,
            "expected_shape": list(expected),
            "array_shape": list(x_eval.shape),
        })

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "input_info.json").write_text(json.dumps(input_info, indent=2), encoding="utf-8")

    summary = []
    for cond in args.conditions:
        parts = cond.lower().split("_")
        ablate = []
        for m in ALL_MODALITIES:
            if m in parts:
                ablate.append(m)
        if "img" in parts:
            ablate.append("image")

        xs = []
        for arr, info in zip(base_arrays, input_info):
            if cond != "normal" and info["modality"] in ablate:
                xs.append(np.zeros_like(arr))
            else:
                xs.append(arr)

        x_input = xs if len(xs) > 1 else xs[0]
        metrics = evaluate_condition(model, x_input, y_true, labels, out_root / cond, cond, args.batch)
        metrics["ablated"] = ",".join(sorted(set(ablate)))
        summary.append(metrics)

    keys = sorted({k for row in summary for k in row.keys()})
    with (out_root / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in summary:
            w.writerow(row)

    (out_root / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[DONE]", out_root)


if __name__ == "__main__":
    main()
