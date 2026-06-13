#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train one OnlyFeet task model for one fold and evaluate it on the held-out group.

Dataset convention:
  data_dir/regular/dataset_train.npz  -> training pool for this fold
  data_dir/regular/dataset_val.npz    -> held-out test group for this fold
  data_dir/regular/dataset_card.json

Important protocol detail:
  dataset_val.npz is not used for checkpoint selection. This script creates an
  internal folder-level validation split from dataset_train.npz.

Supported tasks:
  activity: walk / standing / sitting, label y_act
  surface : walking-only surface, label y_env, filtered to y_act == walk

Supported model configurations:
  single modality: --fusion single --modalities imu
  concat fusion  : --fusion concat  --modalities image,audio
  gated fusion   : --fusion gated   --modalities image,audio,imu
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALL_MODALITIES = ["imu", "tof", "mag", "audio", "image"]
MOD_KEYS = {
    "imu": "imu_win",
    "tof": "tof_win",
    "mag": "mag_win",
    "audio": "audio_win",
    "image": "img_win",
}


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"[INFO] GPUs: {gpus}")
    else:
        print("[WARN] No GPU detected")


def ensure_audio_channel(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.ndim == 3:
        x = x[..., None]
    return x


def ensure_image_float(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.size and np.nanmax(x) > 2.0:
        x = x / 255.0
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def add_imu_magnitudes(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.shape[-1] == 8:
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


def load_card(data_dir: Path) -> dict:
    for p in [data_dir / "regular" / "dataset_card.json", data_dir / "dataset_card.json"]:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"dataset_card.json not found below {data_dir}")


def load_npz_pair(data_dir: Path) -> Tuple[np.lib.npyio.NpzFile, np.lib.npyio.NpzFile, dict]:
    regular = data_dir / "regular" if (data_dir / "regular").exists() else data_dir
    train_path = regular / "dataset_train.npz"
    test_path = regular / "dataset_val.npz"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing dataset_train.npz or dataset_val.npz in {regular}")
    return np.load(train_path, allow_pickle=True), np.load(test_path, allow_pickle=True), load_card(data_dir)


def load_arrays(npz, modalities: List[str]) -> Dict[str, np.ndarray]:
    x = {}
    for m in modalities:
        if MOD_KEYS[m] not in npz.files:
            raise KeyError(f"Missing {MOD_KEYS[m]} in NPZ")
        arr = npz[MOD_KEYS[m]]
        if m == "imu":
            arr = add_imu_magnitudes(arr)
        elif m == "audio":
            arr = ensure_audio_channel(arr)
        elif m == "image":
            arr = ensure_image_float(arr)
        else:
            arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        x[m] = arr.astype(np.float32)
    return x


def task_labels_and_mask(npz, card: dict, task: str):
    y_act = npz["y_act"].astype(np.int64)
    y_env = npz["y_env"].astype(np.int64)
    act_names = list(card.get("activity_classes", [str(i) for i in range(int(y_act.max()) + 1)]))
    env_names = list(card.get("env_classes", [str(i) for i in range(int(y_env[y_env >= 0].max()) + 1)]))

    if task == "activity":
        valid = y_act >= 0
        return y_act, valid, act_names

    if task == "surface":
        if "walk" in act_names:
            walk_id = act_names.index("walk")
            valid = (y_env >= 0) & (y_act == walk_id)
        else:
            valid = y_env >= 0
        # Respect explicit surface mask if present.
        for k in ["sw_env", "sw_env_mask", "env_mask", "surface_mask", "y_env_mask"]:
            if k in npz.files and len(npz[k]) == len(y_env):
                valid = valid & (npz[k] > 0)
        return y_env, valid, env_names

    raise ValueError(f"Unknown task: {task}")


def get_trace(npz, valid: np.ndarray) -> dict:
    n = len(valid)
    folders = np.array([str(x) for x in npz["folder"]], dtype=object) if "folder" in npz.files else np.array([f"idx_{i}" for i in range(n)], dtype=object)
    start_ms = npz["start_ms"].astype(float) if "start_ms" in npz.files else np.arange(n, dtype=float)
    return {"folder": folders[valid], "start_ms": start_ms[valid]}


def split_internal_by_folder(folders: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.array(sorted(set(map(str, folders))))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_frac))) if len(unique) > 1 else 1
    val_folders = set(unique[:n_val].tolist())
    val_mask = np.array([str(f) in val_folders for f in folders], dtype=bool)
    train_mask = ~val_mask
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        # Fallback sample-level split, only if folder split impossible.
        idx = np.arange(len(folders))
        rng.shuffle(idx)
        n_val_s = max(1, int(round(len(idx) * val_frac)))
        val_idx = set(idx[:n_val_s].tolist())
        val_mask = np.array([i in val_idx for i in range(len(folders))])
        train_mask = ~val_mask
    return train_mask, val_mask


def normalize_train_val_test(x_train_pool, train_mask, val_mask, x_test):
    out_tr, out_val, out_test, stats = {}, {}, {}, {}
    for m, arr in x_train_pool.items():
        tr = arr[train_mask]
        va = arr[val_mask]
        te = x_test[m]
        axes = tuple(range(tr.ndim - 1))
        mean = tr.mean(axis=axes, keepdims=True)
        std = tr.std(axis=axes, keepdims=True) + 1e-6
        out_tr[m] = np.nan_to_num((tr - mean) / std).astype(np.float32)
        out_val[m] = np.nan_to_num((va - mean) / std).astype(np.float32)
        out_test[m] = np.nan_to_num((te - mean) / std).astype(np.float32)
        stats[m] = {"mean_shape": list(mean.shape), "std_shape": list(std.shape), "train_mean_global": float(tr.mean()), "train_std_global": float(tr.std())}
    return out_tr, out_val, out_test, stats


def to_model_input(x: Dict[str, np.ndarray], modalities: List[str]):
    return {m: x[m] for m in modalities}


def dense_project(x, dim: int, name: str, dropout: float):
    x = layers.Dense(dim, activation="relu", name=f"{name}_proj_dense")(x)
    x = layers.BatchNormalization(name=f"{name}_proj_bn")(x)
    x = layers.Dropout(dropout, name=f"{name}_proj_dropout")(x)
    return x


def encoder(m: str, input_shape, feature_dim: int, dropout: float):
    inp = layers.Input(shape=input_shape, name=m)
    if m == "imu":
        x = layers.Conv1D(64, 3, padding="same", activation="relu", name="imu_conv1")(inp)
        x = layers.BatchNormalization(name="imu_bn1")(x)
        x = layers.MaxPooling1D(2, name="imu_pool1")(x)
        x = layers.Conv1D(128, 3, padding="same", activation="relu", name="imu_conv2")(x)
        x = layers.BatchNormalization(name="imu_bn2")(x)
        x = layers.GRU(64, dropout=min(dropout, 0.5), name="imu_gru")(x)
        feat = dense_project(x, feature_dim, "imu", dropout)
    elif m in ["tof", "mag"]:
        x = layers.Conv1D(32, 3, padding="same", activation="relu", name=f"{m}_conv1")(inp)
        x = layers.BatchNormalization(name=f"{m}_bn1")(x)
        x = layers.MaxPooling1D(2, name=f"{m}_pool1")(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu", name=f"{m}_conv2")(x)
        x = layers.GlobalAveragePooling1D(name=f"{m}_gap")(x)
        feat = dense_project(x, feature_dim, m, dropout)
    elif m in ["audio", "image"]:
        x = layers.Conv2D(32, 3, padding="same", activation="relu", name=f"{m}_conv1")(inp)
        x = layers.BatchNormalization(name=f"{m}_bn1")(x)
        x = layers.MaxPooling2D(2, name=f"{m}_pool1")(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu", name=f"{m}_conv2")(x)
        x = layers.BatchNormalization(name=f"{m}_bn2")(x)
        x = layers.MaxPooling2D(2, name=f"{m}_pool2")(x)
        if m == "image":
            x = layers.Conv2D(128, 3, padding="same", activation="relu", name=f"{m}_conv3")(x)
            x = layers.BatchNormalization(name=f"{m}_bn3")(x)
        x = layers.GlobalAveragePooling2D(name=f"{m}_gap")(x)
        feat = dense_project(x, feature_dim, m, dropout)
    else:
        raise ValueError(m)
    return inp, feat


def build_model(input_shapes: Dict[str, tuple], modalities: List[str], fusion: str, n_classes: int, feature_dim: int, dropout: float, lr: float, label_smoothing: float):
    inputs, feats = [], []
    for m in modalities:
        inp, feat = encoder(m, input_shapes[m], feature_dim, dropout)
        inputs.append(inp)
        feats.append(feat)

    if fusion == "single":
        if len(feats) != 1:
            raise ValueError("fusion=single requires exactly one modality")
        z = feats[0]
    elif fusion == "concat":
        z = layers.Concatenate(name="concat_fusion")(feats)
        z = layers.Dense(feature_dim, activation="relu", name="fusion_dense")(z)
        z = layers.BatchNormalization(name="fusion_bn")(z)
        z = layers.Dropout(dropout, name="fusion_dropout")(z)
    elif fusion == "gated":
        ctx = layers.Concatenate(name="gate_context")(feats)
        w = layers.Dense(max(32, feature_dim // 2), activation="relu", name="gate_hidden")(ctx)
        w = layers.Dense(len(feats), activation="softmax", name="gate_weights")(w)
        reshaped = [layers.Reshape((1, feature_dim), name=f"gate_reshape_{i}")(f) for i, f in enumerate(feats)]
        stack = layers.Concatenate(axis=1, name="feature_stack")(reshaped)
        z = layers.Dot(axes=(1, 1), name="gated_weighted_sum")([w, stack])
        z = layers.Dense(feature_dim, activation="relu", name="gated_fusion_dense")(z)
        z = layers.BatchNormalization(name="gated_fusion_bn")(z)
        z = layers.Dropout(dropout, name="gated_fusion_dropout")(z)
    else:
        raise ValueError(f"Unsupported fusion: {fusion}")

    z = layers.Dense(128, activation="relu", name="head_dense1")(z)
    z = layers.Dropout(dropout, name="head_dropout1")(z)
    out = layers.Dense(n_classes, activation="softmax", name="output")(z)
    model = keras.Model(inputs=inputs, outputs=out, name="OnlyFeet_Task_CV_Model")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=["accuracy"],
    )
    return model


def class_sample_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    present = np.unique(y)
    weights = np.ones(n_classes, dtype=np.float32)
    cw = compute_class_weight(class_weight="balanced", classes=present, y=y)
    for c, w in zip(present, cw):
        weights[int(c)] = float(w)
    return np.array([weights[int(v)] for v in y], dtype=np.float32)


def save_cm(cm, labels, path, title, normalize=False):
    data = cm.astype(float)

    if normalize:
        row_sum = data.sum(axis=1, keepdims=True)
        data = np.divide(
            data,
            row_sum,
            out=np.zeros_like(data, dtype=float),
            where=row_sum != 0,
        )

    fig, ax = plt.subplots(
        figsize=(max(5, len(labels) * 1.2), max(4, len(labels) * 1.0))
    )
    im = ax.imshow(data)
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if normalize:
                text_value = f"{float(data[i, j]):.2f}"
            else:
                text_value = str(int(round(float(data[i, j]))))
            ax.text(j, i, text_value, ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)



def eval_metrics(y_true, y_pred, labels):
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n": int(len(y_true)),
    }


def save_eval_outputs(out_dir: Path, prefix: str, y_true, y_pred, proba, labels, trace: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = eval_metrics(y_true, y_pred, labels)
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / f"{prefix}_classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=labels, zero_division=0, digits=4), encoding="utf-8"
    )
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0, output_dict=True)
    (out_dir / f"{prefix}_classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(labels))), zero_division=0)
    with (out_dir / f"{prefix}_per_class.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for row in zip(labels, p, r, f1, support):
            w.writerow([row[0], float(row[1]), float(row[2]), float(row[3]), int(row[4])])
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    np.save(out_dir / f"{prefix}_confusion_matrix.npy", cm)
    np.savetxt(out_dir / f"{prefix}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    save_cm(cm, labels, out_dir / f"{prefix}_confusion_matrix.png", f"{prefix} confusion matrix", normalize=False)
    save_cm(cm, labels, out_dir / f"{prefix}_confusion_matrix_normalized.png", f"{prefix} normalized confusion matrix", normalize=True)
    with (out_dir / f"{prefix}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "folder", "start_ms", "y_true", "y_pred", "y_true_name", "y_pred_name", "correct"])
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            folder = trace["folder"][i] if trace and "folder" in trace else ""
            start_ms = trace["start_ms"][i] if trace and "start_ms" in trace else ""
            w.writerow([i, folder, start_ms, int(yt), int(yp), labels[int(yt)], labels[int(yp)], int(yt == yp)])
    return metrics


def nonoverlap_indices(trace: dict, min_gap_ms: float) -> np.ndarray:
    keep = []
    by_folder = {}
    for i, f in enumerate(trace["folder"]):
        by_folder.setdefault(str(f), []).append(i)
    for f, idxs in by_folder.items():
        idxs = sorted(idxs, key=lambda i: float(trace["start_ms"][i]))
        last = -1e18
        for i in idxs:
            t = float(trace["start_ms"][i])
            if t - last >= min_gap_ms:
                keep.append(i)
                last = t
    return np.array(sorted(keep), dtype=int)


def folder_majority(y_true, y_pred, folders):
    out_true, out_pred = [], []
    for f in sorted(set(map(str, folders))):
        idx = np.array([str(x) == f for x in folders])
        yt = y_true[idx]
        yp = y_pred[idx]
        out_true.append(int(np.bincount(yt).argmax()))
        out_pred.append(int(np.bincount(yp).argmax()))
    return np.array(out_true, dtype=int), np.array(out_pred, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--task", required=True, choices=["activity", "surface"])
    ap.add_argument("--modalities", required=True, help="Comma-separated subset of imu,tof,mag,audio,image")
    ap.add_argument("--fusion", required=True, choices=["single", "concat", "gated"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--feature_dim", type=int, default=96)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--internal_val_frac", type=float, default=0.20)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--shuffle_train_labels", action="store_true", help="Label-shuffle sanity check")
    ap.add_argument("--no_zero_input_eval", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    setup_gpu()

    modalities = [m.strip().lower() for m in args.modalities.split(",") if m.strip()]
    bad = [m for m in modalities if m not in ALL_MODALITIES]
    if bad:
        raise ValueError(f"Unsupported modalities: {bad}")
    if args.fusion == "single" and len(modalities) != 1:
        raise ValueError("fusion=single requires exactly one modality")
    if args.fusion in ["concat", "gated"] and len(modalities) < 2:
        raise ValueError("concat/gated fusion requires at least two modalities")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_npz, te_npz, card = load_npz_pair(data_dir)
    x_pool_all = load_arrays(tr_npz, modalities)
    x_test_all = load_arrays(te_npz, modalities)
    y_pool_raw, valid_pool, labels = task_labels_and_mask(tr_npz, card, args.task)
    y_test_raw, valid_test, labels_test = task_labels_and_mask(te_npz, card, args.task)
    if labels != labels_test:
        print("[WARN] train/test labels differ; using training labels")
    n_classes = len(labels)

    folders_pool = np.array([str(x) for x in tr_npz["folder"]], dtype=object) if "folder" in tr_npz.files else np.array([f"idx_{i}" for i in range(len(y_pool_raw))], dtype=object)
    trace_test = get_trace(te_npz, valid_test)

    y_pool = y_pool_raw[valid_pool]
    y_test = y_test_raw[valid_test]
    x_pool = {m: arr[valid_pool] for m, arr in x_pool_all.items()}
    x_test = {m: arr[valid_test] for m, arr in x_test_all.items()}
    folders_pool_valid = folders_pool[valid_pool]

    train_mask, intval_mask = split_internal_by_folder(folders_pool_valid, args.internal_val_frac, args.seed)
    y_train = y_pool[train_mask].copy()
    y_intval = y_pool[intval_mask].copy()

    if args.shuffle_train_labels:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(y_train)

    x_train, x_intval, x_test_norm, norm_stats = normalize_train_val_test(x_pool, train_mask, intval_mask, x_test)

    y_train_oh = keras.utils.to_categorical(y_train, n_classes)
    y_intval_oh = keras.utils.to_categorical(y_intval, n_classes)
    sw_train = class_sample_weights(y_train, n_classes)

    input_shapes = {m: x_train[m].shape[1:] for m in modalities}
    model = build_model(input_shapes, modalities, args.fusion, n_classes, args.feature_dim, args.dropout, args.lr, args.label_smoothing)
    with (out_dir / "model_summary.txt").open("w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))

    config = {
        "fold": args.fold,
        "task": args.task,
        "modalities": modalities,
        "fusion": args.fusion,
        "seed": args.seed,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "classes": labels,
        "n_classes": n_classes,
        "n_train_internal": int(len(y_train)),
        "n_internal_val": int(len(y_intval)),
        "n_test_heldout": int(len(y_test)),
        "input_shapes": {k: list(v) for k, v in input_shapes.items()},
        "shuffle_train_labels": bool(args.shuffle_train_labels),
        "args": vars(args),
        "normalization_stats_summary": norm_stats,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    callbacks = [
        keras.callbacks.CSVLogger(str(out_dir / "training_log.csv")),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=max(3, args.patience // 3), min_lr=1e-6, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(str(out_dir / "best_model.h5"), monitor="val_loss", mode="min", save_best_only=True, verbose=1),
    ]

    print("[INFO] Training", config)
    hist = model.fit(
        to_model_input(x_train, modalities), y_train_oh,
        sample_weight=sw_train,
        validation_data=(to_model_input(x_intval, modalities), y_intval_oh),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        verbose=2,
    )
    (out_dir / "history.json").write_text(json.dumps({k: [float(vv) for vv in v] for k, v in hist.history.items()}, indent=2), encoding="utf-8")

    best = keras.models.load_model(out_dir / "best_model.h5", compile=False)
    proba = best.predict(to_model_input(x_test_norm, modalities), batch_size=args.batch, verbose=1)
    pred = np.argmax(proba, axis=1)
    main_metrics = save_eval_outputs(out_dir, "test", y_test, pred, proba, labels, trace_test)

    # Non-overlapping-window diagnostic.
    keep = nonoverlap_indices(trace_test, min_gap_ms=2000.0)
    nonoverlap_metrics = {}
    if len(keep) > 0:
        nonoverlap_metrics = save_eval_outputs(out_dir, "test_nonoverlap", y_test[keep], pred[keep], proba[keep], labels, {"folder": trace_test["folder"][keep], "start_ms": trace_test["start_ms"][keep]})

    # Folder-level majority vote diagnostic.
    y_f_true, y_f_pred = folder_majority(y_test, pred, trace_test["folder"])
    folder_metrics = eval_metrics(y_f_true, y_f_pred, labels)
    (out_dir / "folder_majority_metrics.json").write_text(json.dumps(folder_metrics, indent=2), encoding="utf-8")

    # Majority-class baseline.
    maj = int(np.bincount(y_test, minlength=n_classes).argmax())
    majority_metrics = eval_metrics(y_test, np.full_like(y_test, maj), labels)
    majority_metrics["majority_class"] = labels[maj]
    (out_dir / "majority_baseline_metrics.json").write_text(json.dumps(majority_metrics, indent=2), encoding="utf-8")

    zero_metrics = {}
    if not args.no_zero_input_eval and len(modalities) >= 2:
        for m in modalities:
            x_zero = {k: np.array(v, copy=True) for k, v in x_test_norm.items()}
            x_zero[m] = np.zeros_like(x_zero[m])
            p0 = best.predict(to_model_input(x_zero, modalities), batch_size=args.batch, verbose=0)
            pred0 = np.argmax(p0, axis=1)
            zero_metrics[f"zero_{m}"] = eval_metrics(y_test, pred0, labels)
        (out_dir / "zero_input_metrics.json").write_text(json.dumps(zero_metrics, indent=2), encoding="utf-8")

    full_summary = {
        "config": config,
        "test": main_metrics,
        "nonoverlap": nonoverlap_metrics,
        "folder_majority": folder_metrics,
        "majority_baseline": majority_metrics,
        "zero_input": zero_metrics,
        "model_params": int(best.count_params()),
        "best_model_size_mb": float((out_dir / "best_model.h5").stat().st_size / 1024 / 1024),
    }
    (out_dir / "summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    print("[DONE]", out_dir)
    print(json.dumps({"test": main_metrics, "nonoverlap": nonoverlap_metrics, "folder": folder_metrics}, indent=2))


if __name__ == "__main__":
    main()
