#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_m14_task_model.py

Universal M14 task-specialist trainer for OnlyFeet.

Supports:
- Task A: activity recognition (walk / standing / sitting)
- Task B: surface recognition (asphalt / PVC / sand / gravel / grass)
- Single-modality baselines
- Multimodal concat fusion
- Multimodal gated fusion

Expected dataset layout:
  datasets_m14_rgb64_stage1/activity/dataset_train.npz
  datasets_m14_rgb64_stage1/activity/dataset_val.npz
  datasets_m14_rgb64_stage1/activity/dataset_card.json

  datasets_m14_rgb64_stage1/surface/dataset_train.npz
  datasets_m14_rgb64_stage1/surface/dataset_val.npz
  datasets_m14_rgb64_stage1/surface/dataset_card.json

  datasets_m14_rgb64_stage2/activity/dataset_train.npz
  datasets_m14_rgb64_stage2/activity/dataset_test.npz
  datasets_m14_rgb64_stage2/activity/dataset_card.json

  datasets_m14_rgb64_stage2/surface/dataset_train.npz
  datasets_m14_rgb64_stage2/surface/dataset_test.npz
  datasets_m14_rgb64_stage2/surface/dataset_card.json
"""

import argparse
import csv
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks, regularizers
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALL_MODALITIES = ["imu", "tof", "mag", "audio", "image"]
NPZ_KEYS = {
    "imu": "imu_win",
    "tof": "tof_win",
    "mag": "mag_win",
    "audio": "audio_win",
    "image": "img_win",
}


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] GPU detected: {len(gpus)}")
        except RuntimeError as e:
            print(f"[WARN] Could not set GPU memory growth: {e}")
    else:
        print("[WARN] No GPU detected; training will run on CPU.")


def parse_modalities(s: str):
    mods = [x.strip().lower() for x in str(s).split(",") if x.strip()]
    if not mods:
        raise ValueError("No modalities selected.")
    bad = [m for m in mods if m not in ALL_MODALITIES]
    if bad:
        raise ValueError(f"Unsupported modalities: {bad}. Valid: {ALL_MODALITIES}")
    return mods


def load_card(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if x.ndim != 3 or x.shape[-1] < 6:
        return x
    if x.shape[-1] >= 8:
        return x
    acc = x[..., 0:3]
    gyro = x[..., 3:6]
    acc_mag = np.sqrt(np.sum(acc ** 2, axis=-1, keepdims=True))
    gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=-1, keepdims=True))
    return np.concatenate([x, acc_mag, gyro_mag], axis=-1).astype(np.float32)


def normalize_train_eval(x_tr: np.ndarray, x_ev: np.ndarray, eps: float = 1e-6):
    # Normalize feature-wise using training statistics only.
    axes = tuple(range(x_tr.ndim - 1))
    mean = x_tr.mean(axis=axes, keepdims=True)
    std = x_tr.std(axis=axes, keepdims=True) + eps
    x_tr = (x_tr - mean) / std
    x_ev = (x_ev - mean) / std
    x_tr = np.nan_to_num(x_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x_ev = np.nan_to_num(x_ev, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x_tr, x_ev


def load_x(npz, modality: str, imu_magnitude: bool = True) -> np.ndarray:
    key = NPZ_KEYS[modality]
    if key not in npz.files:
        raise KeyError(f"Missing {key} in npz. Available keys: {npz.files}")
    x = npz[key]
    if modality == "audio":
        return ensure_audio_channel(x)
    if modality == "image":
        return ensure_image_float(x)
    if modality == "imu" and imu_magnitude:
        return add_imu_magnitudes(x)
    return np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def get_labels(npz, card: dict, task: str):
    if task == "activity":
        y = npz["y_act"].astype(np.int64)
        names = card.get("activity_classes", card.get("act_classes"))
    elif task == "surface":
        y = npz["y_env"].astype(np.int64)
        names = card.get("env_classes", card.get("surface_classes", card.get("environment_classes")))
    else:
        raise ValueError(f"Unsupported task: {task}")
    if names is None:
        names = [str(i) for i in range(int(y[y >= 0].max()) + 1)]
    valid = y >= 0
    return y, valid, list(names)


def make_sample_weights(y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    mapping = {int(c): float(w) for c, w in zip(classes, weights)}
    return np.array([mapping[int(v)] for v in y], dtype=np.float32)


def encoder_imu(inp, feature_dim: int, dropout: float, l2_value: float):
    reg = regularizers.l2(l2_value) if l2_value > 0 else None
    x = layers.Conv1D(64, 5, padding="same", activation="relu", kernel_regularizer=reg, name="imu_conv1")(inp)
    x = layers.BatchNormalization(name="imu_bn1")(x)
    x = layers.MaxPooling1D(2, name="imu_pool1")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu", kernel_regularizer=reg, name="imu_conv2")(x)
    x = layers.BatchNormalization(name="imu_bn2")(x)
    x = layers.GRU(64, dropout=min(dropout, 0.4), name="imu_gru")(x)
    x = layers.Dense(feature_dim, activation="relu", kernel_regularizer=reg, name="imu_feat")(x)
    x = layers.Dropout(dropout, name="imu_dropout")(x)
    return x


def encoder_timeseries(inp, prefix: str, feature_dim: int, dropout: float, l2_value: float):
    reg = regularizers.l2(l2_value) if l2_value > 0 else None
    x = layers.Conv1D(48, 5, padding="same", activation="relu", kernel_regularizer=reg, name=f"{prefix}_conv1")(inp)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.MaxPooling1D(2, name=f"{prefix}_pool1")(x)
    x = layers.Conv1D(96, 3, padding="same", activation="relu", kernel_regularizer=reg, name=f"{prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.GlobalAveragePooling1D(name=f"{prefix}_gap")(x)
    x = layers.Dense(feature_dim, activation="relu", kernel_regularizer=reg, name=f"{prefix}_feat")(x)
    x = layers.Dropout(dropout, name=f"{prefix}_dropout")(x)
    return x


def encoder_2d(inp, prefix: str, feature_dim: int, dropout: float, l2_value: float):
    reg = regularizers.l2(l2_value) if l2_value > 0 else None
    x = layers.Conv2D(32, 3, padding="same", activation="relu", kernel_regularizer=reg, name=f"{prefix}_conv1")(inp)
    x = layers.BatchNormalization(name=f"{prefix}_bn1")(x)
    x = layers.MaxPooling2D(2, name=f"{prefix}_pool1")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", kernel_regularizer=reg, name=f"{prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn2")(x)
    x = layers.MaxPooling2D(2, name=f"{prefix}_pool2")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu", kernel_regularizer=reg, name=f"{prefix}_conv3")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn3")(x)
    x = layers.GlobalAveragePooling2D(name=f"{prefix}_gap")(x)
    x = layers.Dense(feature_dim, activation="relu", kernel_regularizer=reg, name=f"{prefix}_feat")(x)
    x = layers.Dropout(dropout, name=f"{prefix}_dropout")(x)
    return x


def build_encoder(modality: str, input_shape, feature_dim: int, dropout: float, l2_value: float):
    inp = layers.Input(shape=input_shape, name=modality)
    if modality == "imu":
        feat = encoder_imu(inp, feature_dim, dropout, l2_value)
    elif modality in ["tof", "mag"]:
        feat = encoder_timeseries(inp, modality, feature_dim, dropout, l2_value)
    elif modality in ["audio", "image"]:
        feat = encoder_2d(inp, modality, feature_dim, dropout, l2_value)
    else:
        raise ValueError(modality)
    return inp, feat


def gated_fusion(features, modalities, feature_dim: int, dropout: float, l2_value: float):
    reg = regularizers.l2(l2_value) if l2_value > 0 else None
    context = layers.Concatenate(name="gate_context")(features)
    hidden = layers.Dense(64, activation="relu", kernel_regularizer=reg, name="gate_hidden")(context)
    weights = layers.Dense(len(features), activation="softmax", name="modality_weights")(hidden)
    reshaped = [layers.Reshape((1, feature_dim), name=f"reshape_{m}")(f) for m, f in zip(modalities, features)]
    stack = layers.Concatenate(axis=1, name="feature_stack")(reshaped)
    fused = layers.Dot(axes=(1, 1), name="weighted_sum")([weights, stack])
    fused = layers.Dense(feature_dim, activation="relu", kernel_regularizer=reg, name="fused_dense")(fused)
    fused = layers.BatchNormalization(name="fused_bn")(fused)
    fused = layers.Dropout(dropout, name="fused_dropout")(fused)
    return fused, weights


def build_model(input_shapes: dict, modalities: list, n_classes: int, fusion: str, feature_dim: int, dropout: float, l2_value: float):
    inputs = []
    features = []
    for m in modalities:
        inp, feat = build_encoder(m, input_shapes[m], feature_dim, dropout, l2_value)
        inputs.append(inp)
        features.append(feat)

    if len(features) == 1:
        fused = features[0]
    else:
        if fusion == "concat":
            fused = layers.Concatenate(name="concat_fusion")(features)
            fused = layers.Dense(feature_dim, activation="relu", name="concat_fused_dense")(fused)
            fused = layers.BatchNormalization(name="concat_fused_bn")(fused)
            fused = layers.Dropout(dropout, name="concat_fused_dropout")(fused)
        elif fusion == "gated":
            fused, _ = gated_fusion(features, modalities, feature_dim, dropout, l2_value)
        else:
            raise ValueError("fusion must be concat or gated for multimodal inputs")

    x = layers.Dense(128, activation="relu", name="head_dense1")(fused)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(dropout, name="head_dropout1")(x)
    x = layers.Dense(64, activation="relu", name="head_dense2")(x)
    x = layers.Dropout(dropout * 0.75, name="head_dropout2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    name = f"M14_{'_'.join(modalities)}_{fusion}_classifier"
    return Model(inputs=inputs if len(inputs) > 1 else inputs[0], outputs=out, name=name)


def save_confusion_matrix(cm, labels, path: Path, title: str, normalize: bool = False):
    data = cm.astype(float)
    if normalize:
        denom = np.maximum(data.sum(axis=1, keepdims=True), 1.0)
        data = data / denom
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), max(5, len(labels) * 0.9)))
    im = ax.imshow(data, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = ".2f" if normalize else ".0f"
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, format(data[i, j], fmt), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate_and_save(model, x_eval, y_true, class_names, out_dir: Path, batch: int, prefix: str = "eval"):
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_prob = model.predict(x_eval, batch_size=batch, verbose=1)
    y_pred = np.argmax(pred_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "n_eval": int(len(y_true)),
        "classes": class_names,
        "model_params": int(model.count_params()),
    }
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    (out_dir / f"{prefix}_classification_report.txt").write_text(report_txt, encoding="utf-8")
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0, output_dict=True)
    (out_dir / f"{prefix}_classification_report.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")

    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    with (out_dir / f"{prefix}_per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for name, pp, rr, ff, ss in zip(class_names, p, r, f1, support):
            w.writerow([name, float(pp), float(rr), float(ff), int(ss)])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    np.save(out_dir / f"{prefix}_confusion_matrix.npy", cm)
    np.savetxt(out_dir / f"{prefix}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    save_confusion_matrix(cm, class_names, out_dir / f"{prefix}_confusion_matrix.png", f"{prefix} confusion matrix", normalize=False)
    save_confusion_matrix(cm, class_names, out_dir / f"{prefix}_confusion_matrix_normalized.png", f"{prefix} normalized confusion matrix", normalize=True)

    np.savez_compressed(out_dir / f"{prefix}_predictions.npz", y_true=y_true, y_pred=y_pred, pred_prob=pred_prob)
    with (out_dir / f"{prefix}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "y_true", "y_pred", "y_true_name", "y_pred_name", "correct"])
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            w.writerow([i, int(yt), int(yp), class_names[int(yt)], class_names[int(yp)], int(yt == yp)])
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["activity", "surface"])
    ap.add_argument("--modalities", required=True, help="Comma-separated: imu,image,audio,tof,mag")
    ap.add_argument("--fusion", default="gated", choices=["single", "concat", "gated"])
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--eval_npz", required=True, help="Validation npz for Stage 1 or test npz for Stage 2")
    ap.add_argument("--card", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--feature_dim", type=int, default=128)
    ap.add_argument("--patience", type=int, default=14)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_imu_magnitude", action="store_true")
    ap.add_argument("--monitor", default="val_loss")
    args = ap.parse_args()

    set_seed(args.seed)
    setup_gpu()

    modalities = parse_modalities(args.modalities)
    if len(modalities) == 1:
        args.fusion = "single"
    elif args.fusion == "single":
        raise ValueError("fusion=single is only valid for exactly one modality.")

    train_npz_path = Path(args.train_npz)
    eval_npz_path = Path(args.eval_npz)
    card_path = Path(args.card)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    card = load_card(card_path)
    tr = np.load(train_npz_path, allow_pickle=True)
    ev = np.load(eval_npz_path, allow_pickle=True)

    x_tr_dict = {}
    x_ev_dict = {}
    for m in modalities:
        xtr = load_x(tr, m, imu_magnitude=not args.no_imu_magnitude)
        xev = load_x(ev, m, imu_magnitude=not args.no_imu_magnitude)
        xtr, xev = normalize_train_eval(xtr, xev)
        x_tr_dict[m] = xtr
        x_ev_dict[m] = xev

    y_tr_all, valid_tr, class_names = get_labels(tr, card, args.task)
    y_ev_all, valid_ev, class_names_eval = get_labels(ev, card, args.task)
    if class_names != class_names_eval:
        print("[WARN] Train/eval class names differ. Using train card class names.")
    y_tr = y_tr_all[valid_tr]
    y_ev = y_ev_all[valid_ev]
    x_tr = {m: x_tr_dict[m][valid_tr] for m in modalities}
    x_ev = {m: x_ev_dict[m][valid_ev] for m in modalities}

    n_classes = len(class_names)
    input_shapes = {m: x_tr[m].shape[1:] for m in modalities}
    print("[INFO] task:", args.task)
    print("[INFO] modalities:", modalities)
    print("[INFO] fusion:", args.fusion)
    print("[INFO] train samples:", len(y_tr), "eval samples:", len(y_ev))
    print("[INFO] class names:", class_names)
    print("[INFO] train bincount:", np.bincount(y_tr, minlength=n_classes).tolist())
    print("[INFO] eval  bincount:", np.bincount(y_ev, minlength=n_classes).tolist())
    print("[INFO] input shapes:", {k: list(v) for k, v in input_shapes.items()})

    model = build_model(
        input_shapes=input_shapes,
        modalities=modalities,
        n_classes=n_classes,
        fusion=args.fusion,
        feature_dim=args.feature_dim,
        dropout=args.dropout,
        l2_value=args.l2,
    )
    use_onehot = args.label_smoothing > 0
    if use_onehot:
        y_tr_fit = keras.utils.to_categorical(y_tr, n_classes)
        y_ev_fit = keras.utils.to_categorical(y_ev, n_classes)
        loss_obj = keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        metric_obj = keras.metrics.CategoricalAccuracy(name="accuracy")
    else:
        y_tr_fit = y_tr
        y_ev_fit = y_ev
        loss_obj = keras.losses.SparseCategoricalCrossentropy()
        metric_obj = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_obj,
        metrics=[metric_obj],
    )
    model.summary()

    train_x = [x_tr[m] for m in modalities] if len(modalities) > 1 else x_tr[modalities[0]]
    eval_x = [x_ev[m] for m in modalities] if len(modalities) > 1 else x_ev[modalities[0]]
    sw_tr = make_sample_weights(y_tr)

    config = vars(args).copy()
    config.update({
        "modalities_parsed": modalities,
        "input_shapes": {k: list(v) for k, v in input_shapes.items()},
        "class_names": class_names,
        "train_npz": str(train_npz_path),
        "eval_npz": str(eval_npz_path),
        "card": str(card_path),
        "train_samples": int(len(y_tr)),
        "eval_samples": int(len(y_ev)),
        "train_bincount": np.bincount(y_tr, minlength=n_classes).astype(int).tolist(),
        "eval_bincount": np.bincount(y_ev, minlength=n_classes).astype(int).tolist(),
    })
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_model_path = out_dir / "best_model.h5"
    cb = [
        callbacks.ModelCheckpoint(str(best_model_path), monitor=args.monitor, mode="min", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor=args.monitor, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(monitor=args.monitor, mode="min", patience=args.patience, restore_best_weights=True, verbose=1),
        callbacks.CSVLogger(str(out_dir / "training_log.csv")),
    ]

    hist = model.fit(
        train_x,
        y_tr_fit,
        validation_data=(eval_x, y_ev_fit),
        sample_weight=sw_tr,
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cb,
        verbose=2,
    )
    model.save(out_dir / "final_model.h5")
    (out_dir / "history.json").write_text(json.dumps({k: [float(x) for x in v] for k, v in hist.history.items()}, indent=2), encoding="utf-8")

    best_model = keras.models.load_model(best_model_path, compile=False)
    metrics = evaluate_and_save(best_model, eval_x, y_ev, class_names, out_dir, args.batch, prefix="eval")
    metrics.update({
        "task": args.task,
        "modalities": modalities,
        "fusion": args.fusion,
        "best_model": str(best_model_path),
        "model_size_mb": float(best_model_path.stat().st_size / 1024 / 1024),
    })
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[DONE]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
