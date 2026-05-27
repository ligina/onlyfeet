#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean-P4 Stage 2 trainer.

This is a companion to 02_train_m14_task_model.py for the two final selected
Stage 2 reruns. It uses only the Stage 2 training NPZ for model.fit validation
and loads the P4 test NPZ only after training for final evaluation.
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from tensorflow import keras
from tensorflow.keras import callbacks


def load_base_module():
    base_path = Path(__file__).with_name("02_train_m14_task_model.py")
    spec = importlib.util.spec_from_file_location("m14_task_model_base", base_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base trainer from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def normalize_from_train(x_train: np.ndarray, *others: np.ndarray, eps: float = 1e-6):
    axes = tuple(range(x_train.ndim - 1))
    mean = x_train.mean(axis=axes, keepdims=True)
    std = x_train.std(axis=axes, keepdims=True) + eps

    def apply(x: np.ndarray) -> np.ndarray:
        x = (x - mean) / std
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    normalized = [apply(x_train)]
    normalized.extend(apply(x) for x in others)
    return normalized, {"mean": mean, "std": std}


def apply_normalization(x: np.ndarray, stats: dict) -> np.ndarray:
    x = (x - stats["mean"]) / stats["std"]
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def to_model_x(x_by_modality: dict, modalities: list):
    if len(modalities) == 1:
        return x_by_modality[modalities[0]]
    return [x_by_modality[m] for m in modalities]


def bincount_list(y: np.ndarray, n_classes: int):
    return np.bincount(y, minlength=n_classes).astype(int).tolist()


def choose_folder_split(y: np.ndarray, folders: np.ndarray, ratio: float, seed: int, n_classes: int):
    unique_folders = np.unique(folders)
    if len(unique_folders) < 2:
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


def choose_sample_split(y: np.ndarray, ratio: float, seed: int):
    indices = np.arange(len(y))
    class_counts = np.bincount(y)
    if np.any(class_counts[class_counts > 0] < 2):
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(y) * ratio)))
        val_idx = np.sort(shuffled[:n_val])
        train_idx = np.sort(shuffled[n_val:])
        return train_idx, val_idx, "random_sample_level_internal_validation_low_class_count"

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, y))
    return train_idx, val_idx, "stratified_sample_level_internal_validation"


def make_internal_validation_split(train_npz, valid_mask: np.ndarray, y: np.ndarray, ratio: float, seed: int, n_classes: int):
    if "folder" in train_npz.files:
        folders = np.asarray(train_npz["folder"], dtype=object)[valid_mask].astype(str)
        folder_split = choose_folder_split(y, folders, ratio, seed, n_classes)
        if folder_split is not None:
            train_idx, val_idx = folder_split
            train_groups = set(folders[train_idx].tolist())
            val_groups = set(folders[val_idx].tolist())
            overlap = sorted(train_groups.intersection(val_groups))
            if overlap:
                raise RuntimeError(f"Internal folder split overlap detected: {overlap[:5]}")
            return {
                "train_idx": np.sort(train_idx),
                "val_idx": np.sort(val_idx),
                "method": "folder_level_internal_validation",
                "group_key": "folder",
                "train_groups": len(train_groups),
                "val_groups": len(val_groups),
                "group_overlap_count": len(overlap),
                "limitation": None,
            }

    train_idx, val_idx, method = choose_sample_split(y, ratio, seed)
    return {
        "train_idx": np.sort(train_idx),
        "val_idx": np.sort(val_idx),
        "method": method,
        "group_key": None,
        "train_groups": None,
        "val_groups": None,
        "group_overlap_count": None,
        "limitation": "No usable folder key was available; internal validation is sample-level.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["activity", "surface"])
    ap.add_argument("--modalities", required=True, help="Comma-separated: imu,image,audio,tof,mag")
    ap.add_argument("--fusion", default="gated", choices=["single", "concat", "gated"])
    ap.add_argument("--train_npz", required=True)
    ap.add_argument("--test_npz", required=True, help="P4 test NPZ; loaded only after training")
    ap.add_argument("--card", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.35)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--feature_dim", type=int, default=128)
    ap.add_argument("--patience", type=int, default=16)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_imu_magnitude", action="store_true")
    ap.add_argument("--monitor", default="val_loss")
    ap.add_argument("--validation_split_ratio", type=float, default=0.15)
    args = ap.parse_args()

    if not (0.0 < args.validation_split_ratio < 0.5):
        raise ValueError("--validation_split_ratio must be between 0 and 0.5")

    base.set_seed(args.seed)
    random.seed(args.seed)
    base.setup_gpu()

    modalities = base.parse_modalities(args.modalities)
    if len(modalities) == 1:
        args.fusion = "single"
    elif args.fusion == "single":
        raise ValueError("fusion=single is only valid for exactly one modality.")

    train_npz_path = Path(args.train_npz)
    test_npz_path = Path(args.test_npz)
    card_path = Path(args.card)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    card = base.load_card(card_path)
    tr = np.load(train_npz_path, allow_pickle=True)

    y_all, valid_tr, class_names = base.get_labels(tr, card, args.task)
    y_all = y_all[valid_tr]
    n_classes = len(class_names)
    split = make_internal_validation_split(
        train_npz=tr,
        valid_mask=valid_tr,
        y=y_all,
        ratio=args.validation_split_ratio,
        seed=args.seed,
        n_classes=n_classes,
    )
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]

    x_train_by_modality = {}
    x_val_by_modality = {}
    norm_stats = {}
    for m in modalities:
        x_all = base.load_x(tr, m, imu_magnitude=not args.no_imu_magnitude)[valid_tr]
        (x_train_norm, x_val_norm), stats = normalize_from_train(x_all[train_idx], x_all[val_idx])
        x_train_by_modality[m] = x_train_norm
        x_val_by_modality[m] = x_val_norm
        norm_stats[m] = stats

    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    input_shapes = {m: x_train_by_modality[m].shape[1:] for m in modalities}

    print("[INFO] CLEAN P4 TRAINING")
    print("[INFO] task:", args.task)
    print("[INFO] modalities:", modalities)
    print("[INFO] fusion:", args.fusion)
    print("[INFO] train_npz:", train_npz_path)
    print("[INFO] test_npz reserved for final eval:", test_npz_path)
    print("[INFO] internal validation method:", split["method"])
    print("[INFO] internal train samples:", len(y_train), "internal val samples:", len(y_val))
    print("[INFO] class names:", class_names)
    print("[INFO] internal train bincount:", bincount_list(y_train, n_classes))
    print("[INFO] internal val bincount:", bincount_list(y_val, n_classes))
    print("[INFO] input shapes:", {k: list(v) for k, v in input_shapes.items()})

    model = base.build_model(
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
        y_train_fit = keras.utils.to_categorical(y_train, n_classes)
        y_val_fit = keras.utils.to_categorical(y_val, n_classes)
        loss_obj = keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)
        metric_obj = keras.metrics.CategoricalAccuracy(name="accuracy")
    else:
        y_train_fit = y_train
        y_val_fit = y_val
        loss_obj = keras.losses.SparseCategoricalCrossentropy()
        metric_obj = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_obj,
        metrics=[metric_obj],
    )
    model.summary()

    train_x = to_model_x(x_train_by_modality, modalities)
    val_x = to_model_x(x_val_by_modality, modalities)
    sw_train = base.make_sample_weights(y_train)

    best_model_path = out_dir / "best_model.h5"
    callback_config = [
        {
            "name": "ModelCheckpoint",
            "monitor": args.monitor,
            "mode": "min",
            "save_best_only": True,
            "path": str(best_model_path),
        },
        {
            "name": "ReduceLROnPlateau",
            "monitor": args.monitor,
            "mode": "min",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6,
        },
        {
            "name": "EarlyStopping",
            "monitor": args.monitor,
            "mode": "min",
            "patience": args.patience,
            "restore_best_weights": True,
        },
        {
            "name": "CSVLogger",
            "path": str(out_dir / "training_log.csv"),
        },
    ]
    cb = [
        callbacks.ModelCheckpoint(str(best_model_path), monitor=args.monitor, mode="min", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor=args.monitor, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(monitor=args.monitor, mode="min", patience=args.patience, restore_best_weights=True, verbose=1),
        callbacks.CSVLogger(str(out_dir / "training_log.csv")),
    ]

    config = vars(args).copy()
    config.update({
        "modalities_parsed": modalities,
        "input_shapes": {k: list(v) for k, v in input_shapes.items()},
        "class_names": class_names,
        "train_npz": str(train_npz_path),
        "internal_val_source": "train_npz only",
        "test_npz": str(test_npz_path),
        "card": str(card_path),
        "train_samples_total_valid": int(len(y_all)),
        "internal_train_samples": int(len(y_train)),
        "internal_val_samples": int(len(y_val)),
        "internal_train_bincount": bincount_list(y_train, n_classes),
        "internal_val_bincount": bincount_list(y_val, n_classes),
        "validation_split_method": split["method"],
        "validation_split_ratio": float(args.validation_split_ratio),
        "validation_group_key": split["group_key"],
        "validation_train_groups": split["train_groups"],
        "validation_val_groups": split["val_groups"],
        "validation_group_overlap_count": split["group_overlap_count"],
        "validation_limitation": split["limitation"],
        "normalization_fit_source": "internal training partition only",
        "monitor": args.monitor,
        "callbacks": callback_config,
        "p4_used_for_training": False,
        "p4_used_for_validation": False,
        "p4_used_for_checkpoint_selection": False,
        "p4_used_for_final_eval_only": True,
    })
    (out_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    hist = model.fit(
        train_x,
        y_train_fit,
        validation_data=(val_x, y_val_fit),
        sample_weight=sw_train,
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

    print("[INFO] Loading P4 test NPZ for final evaluation only:", test_npz_path)
    te = np.load(test_npz_path, allow_pickle=True)
    y_test_all, valid_test, class_names_test = base.get_labels(te, card, args.task)
    if class_names != class_names_test:
        print("[WARN] Train/test class names differ. Using train card class names.")
    y_test = y_test_all[valid_test]
    x_test_by_modality = {}
    for m in modalities:
        x_test_raw = base.load_x(te, m, imu_magnitude=not args.no_imu_magnitude)[valid_test]
        x_test_by_modality[m] = apply_normalization(x_test_raw, norm_stats[m])

    test_x = to_model_x(x_test_by_modality, modalities)
    best_model = keras.models.load_model(best_model_path, compile=False)
    metrics = base.evaluate_and_save(best_model, test_x, y_test, class_names, out_dir, args.batch, prefix="eval")
    metrics.update({
        "task": args.task,
        "modalities": modalities,
        "fusion": args.fusion,
        "best_model": str(best_model_path),
        "model_size_mb": float(best_model_path.stat().st_size / 1024 / 1024),
        "test_npz": str(test_npz_path),
        "p4_used_for_training": False,
        "p4_used_for_validation": False,
        "p4_used_for_checkpoint_selection": False,
        "p4_used_for_final_eval_only": True,
        "validation_split_method": split["method"],
        "validation_split_ratio": float(args.validation_split_ratio),
    })
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[DONE CLEAN P4]")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
