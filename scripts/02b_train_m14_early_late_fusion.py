#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02b_train_m14_early_late_fusion.py

Extra fusion-strategy baseline for M14:
- early_flat: normalize each modality, flatten raw inputs, concatenate, MLP classifier
- late_avg: each modality has its own encoder + softmax head, final prediction is averaged probabilities

This script reuses data loading / normalization / evaluation helpers from 02_train_m14_task_model.py.
Outputs are saved to a separate folder specified by --out_dir.
"""

import argparse
import json
import random
import importlib.util
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks, regularizers
from sklearn.utils.class_weight import compute_class_weight


def import_base():
    p = Path("scripts_m14/02_train_m14_task_model.py")
    spec = importlib.util.spec_from_file_location("base_m14", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed: int):
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
        print(f"[INFO] GPU detected: {len(gpus)}")
    else:
        print("[WARN] No GPU detected.")


def parse_modalities(s):
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def make_sample_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    mp = {int(c): float(w) for c, w in zip(classes, weights)}
    return np.array([mp[int(v)] for v in y], dtype=np.float32)


def build_early_flat_model(input_shapes, modalities, n_classes, dropout=0.45, l2_value=1e-4):
    reg = regularizers.l2(l2_value) if l2_value > 0 else None

    inputs = []
    flats = []

    for m in modalities:
        inp = layers.Input(shape=input_shapes[m], name=m)
        x = layers.Flatten(name=f"{m}_flatten")(inp)
        x = layers.Dense(256, activation="relu", kernel_regularizer=reg, name=f"{m}_proj")(x)
        x = layers.BatchNormalization(name=f"{m}_bn")(x)
        x = layers.Dropout(dropout, name=f"{m}_dropout")(x)
        inputs.append(inp)
        flats.append(x)

    if len(flats) == 1:
        fused = flats[0]
    else:
        fused = layers.Concatenate(name="early_flat_concat")(flats)

    x = layers.Dense(256, activation="relu", kernel_regularizer=reg, name="head_dense1")(fused)
    x = layers.BatchNormalization(name="head_bn1")(x)
    x = layers.Dropout(dropout, name="head_dropout1")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg, name="head_dense2")(x)
    x = layers.Dropout(dropout * 0.75, name="head_dropout2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)

    return Model(inputs=inputs if len(inputs) > 1 else inputs[0], outputs=out,
                 name=f"M14_early_flat_{'_'.join(modalities)}")


def build_late_avg_model(base, input_shapes, modalities, n_classes, feature_dim=128, dropout=0.40, l2_value=1e-4):
    reg = regularizers.l2(l2_value) if l2_value > 0 else None

    inputs = []
    probs = []

    for m in modalities:
        inp, feat = base.build_encoder(m, input_shapes[m], feature_dim, dropout, l2_value)

        x = layers.Dense(128, activation="relu", kernel_regularizer=reg, name=f"{m}_late_dense1")(feat)
        x = layers.BatchNormalization(name=f"{m}_late_bn1")(x)
        x = layers.Dropout(dropout, name=f"{m}_late_dropout1")(x)
        out_m = layers.Dense(n_classes, activation="softmax", name=f"{m}_prob")(x)

        inputs.append(inp)
        probs.append(out_m)

    if len(probs) == 1:
        out = probs[0]
    else:
        out = layers.Average(name="late_average_output")(probs)

    return Model(inputs=inputs if len(inputs) > 1 else inputs[0], outputs=out,
                 name=f"M14_late_avg_{'_'.join(modalities)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["activity", "surface"])
    ap.add_argument("--modalities", required=True)
    ap.add_argument("--fusion_strategy", required=True, choices=["early_flat", "late_avg"])
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--eval_npz", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.45)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--feature_dim", type=int, default=128)
    ap.add_argument("--patience", type=int, default=16)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_imu_magnitude", action="store_true")
    args = ap.parse_args()

    base = import_base()
    set_seed(args.seed)
    setup_gpu()

    modalities = parse_modalities(args.modalities)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    tr = np.load(args.train_npz, allow_pickle=True)
    ev = np.load(args.eval_npz, allow_pickle=True)
    card = base.load_card(Path(args.card))

    x_tr_dict = {}
    x_ev_dict = {}

    for m in modalities:
        xtr = base.load_x(tr, m, imu_magnitude=not args.no_imu_magnitude)
        xev = base.load_x(ev, m, imu_magnitude=not args.no_imu_magnitude)
        xtr, xev = base.normalize_train_eval(xtr, xev)
        x_tr_dict[m] = xtr
        x_ev_dict[m] = xev

    y_tr_all, valid_tr, class_names = base.get_labels(tr, card, args.task)
    y_ev_all, valid_ev, _ = base.get_labels(ev, card, args.task)

    y_tr = y_tr_all[valid_tr]
    y_ev = y_ev_all[valid_ev]

    x_tr = {m: x_tr_dict[m][valid_tr] for m in modalities}
    x_ev = {m: x_ev_dict[m][valid_ev] for m in modalities}

    input_shapes = {m: x_tr[m].shape[1:] for m in modalities}
    n_classes = len(class_names)

    print("[INFO] task:", args.task)
    print("[INFO] modalities:", modalities)
    print("[INFO] fusion_strategy:", args.fusion_strategy)
    print("[INFO] train samples:", len(y_tr), "eval samples:", len(y_ev))
    print("[INFO] classes:", class_names)
    print("[INFO] input_shapes:", {k: list(v) for k, v in input_shapes.items()})
    print("[INFO] train bincount:", np.bincount(y_tr, minlength=n_classes).tolist())
    print("[INFO] eval  bincount:", np.bincount(y_ev, minlength=n_classes).tolist())

    if args.fusion_strategy == "early_flat":
        model = build_early_flat_model(
            input_shapes=input_shapes,
            modalities=modalities,
            n_classes=n_classes,
            dropout=args.dropout,
            l2_value=args.l2,
        )
    else:
        model = build_late_avg_model(
            base=base,
            input_shapes=input_shapes,
            modalities=modalities,
            n_classes=n_classes,
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
    sw = make_sample_weights(y_tr)

    cfg = vars(args).copy()
    cfg.update({
        "modalities_parsed": modalities,
        "input_shapes": {k: list(v) for k, v in input_shapes.items()},
        "class_names": class_names,
        "train_samples": int(len(y_tr)),
        "eval_samples": int(len(y_ev)),
        "train_bincount": np.bincount(y_tr, minlength=n_classes).astype(int).tolist(),
        "eval_bincount": np.bincount(y_ev, minlength=n_classes).astype(int).tolist(),
    })
    (out_dir / "train_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    best_model_path = out_dir / "best_model.h5"

    cb = [
        callbacks.ModelCheckpoint(str(best_model_path), monitor="val_loss", mode="min", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, restore_best_weights=True, verbose=1),
        callbacks.CSVLogger(str(out_dir / "training_log.csv")),
    ]

    hist = model.fit(
        train_x,
        y_tr_fit,
        validation_data=(eval_x, y_ev_fit),
        sample_weight=sw,
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cb,
        verbose=2,
    )

    model.save(out_dir / "final_model.h5")
    (out_dir / "history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in hist.history.items()}, indent=2),
        encoding="utf-8",
    )

    best_model = keras.models.load_model(best_model_path, compile=False)
    metrics = base.evaluate_and_save(best_model, eval_x, y_ev, class_names, out_dir, args.batch, prefix="eval")

    metrics.update({
        "task": args.task,
        "modalities": modalities,
        "fusion": args.fusion_strategy,
        "best_model": str(best_model_path),
        "model_size_mb": float(best_model_path.stat().st_size / 1024 / 1024),
    })

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("[DONE]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
