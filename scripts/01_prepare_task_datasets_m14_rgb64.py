#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_prepare_task_datasets_m14_rgb64.py

OnlyFeet M14 RGB64 dataset preparation.

Purpose
-------
Build the final M14 window-level datasets from raw OnlyFeet folders and
labels_all_m14.csv.

M14 task definition
-------------------
Task A: Regular Activity Recognition
    classes: walk / standing / sitting
    samples: all walk, standing, sitting folders

Task B: Surface / Environment Recognition
    classes: asphalt / PVC / sand / gravel / grass
    samples: walking samples only

Experiment stages
-----------------
Stage 1: model selection / ablation
    train = P1and2
    val   = P3

Stage 2: final held-out test
    train = P1and2 + P3
    test  = P4

Important constraints
---------------------
- Do not create stair datasets.
- Do not include stairs_up / stairs_down.
- Do not infer labels from folder names.
- Always use label_activity and label_env from labels_all_m14.csv.
- Surface task uses walking samples only.
- Keep output directories separate from M13.

Example
-------
cd /home/jovyan/work/bs/final

python scripts_m14/01_prepare_task_datasets_m14_rgb64.py \
  --csv data/labels_all_m14.csv \
  --data_root data \
  --out_stage1 datasets_m14_rgb64_stage1 \
  --out_stage2 datasets_m14_rgb64_stage2
"""

import argparse
import bisect
import json
import math
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

try:
    import librosa
except Exception as e:
    raise ImportError(
        "librosa is required. Install with: pip install librosa soundfile audioread"
    ) from e


# =========================
# Global configuration
# =========================

STEP_MS = 40
WIN_SEC = 2.0
STRIDE_SEC = 1.0

WIN_MS = int(WIN_SEC * 1000)
STRIDE_MS = int(STRIDE_SEC * 1000)

AUDIO_SR = 16000
AUDIO_MELS = 40
AUDIO_STEPS = 40

IMAGE_SIZE = (64, 64)

ACTIVITY_CLASSES = ["walk", "standing", "sitting"]
SURFACE_CLASSES = ["asphalt", "PVC", "sand", "gravel", "grass"]

ALL_MODALITIES = ["imu", "tof", "mag", "audio", "image"]
MASK_ORDER = ["imu", "tof", "mag", "audio", "image"]

ACTIVITY_MAP = {name: i for i, name in enumerate(ACTIVITY_CLASSES)}
SURFACE_MAP = {name: i for i, name in enumerate(SURFACE_CLASSES)}

ACTIVITY_ALIASES = {
    "walk": "walk",
    "walking": "walk",
    "stand": "standing",
    "standing": "standing",
    "sit": "sitting",
    "sitting": "sitting",
}

SURFACE_ALIASES = {
    "asphalt": "asphalt",
    "pvc": "PVC",
    "PVC": "PVC",
    "sand": "sand",
    "gravel": "gravel",
    "grass": "grass",
}

PARTICIPANT_STAGE1_TRAIN = {"P1and2", "p1and2", "P1AND2"}
PARTICIPANT_STAGE1_VAL = {"P3", "p3"}
PARTICIPANT_STAGE2_TRAIN = {"P1and2", "p1and2", "P1AND2", "P3", "p3"}
PARTICIPANT_STAGE2_TEST = {"P4", "p4"}


# =========================
# Utility functions
# =========================

def log(msg: str) -> None:
    print(msg, flush=True)


def slug(s: Any) -> str:
    if s is None:
        return "unknown"
    s = str(s).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s or "unknown"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def parse_last_int_from_stem(path: Path) -> Optional[int]:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return None


def sort_by_last_int(files: List[Path]) -> List[Path]:
    def key(p: Path):
        v = parse_last_int_from_stem(p)
        return v if v is not None else 10**18

    return sorted(files, key=key)


def find_folder_column(df: pd.DataFrame) -> str:
    candidates = [
        "relative_path",
        "folder",
        "folder_path",
        "path",
        "sample_path",
        "sample_folder",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find folder path column. Existing columns: {df.columns.tolist()}")


def find_participant_column(df: pd.DataFrame) -> str:
    candidates = ["participant", "subject", "person", "user", "split_subject"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find participant column. Existing columns: {df.columns.tolist()}")


def resolve_folder(data_root: Path, value: Any) -> Path:
    """
    Resolve folder paths stored in CSV.

    Supports:
      1. absolute path
      2. path relative to data_root
      3. path relative to project root
      4. path starting with data/
      5. folder basename fallback search under data_root
    """
    raw = str(value).strip().replace("\\", "/")
    p = Path(raw)

    if p.is_absolute() and p.exists():
        return p

    project_root = data_root.parent
    candidates: List[Path] = []

    if raw.startswith("data/"):
        candidates.append(project_root / raw)

    candidates.extend([
        data_root / raw,
        project_root / raw,
        Path(raw),
    ])

    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            pass

    # Fallback: search by folder basename. This is slower but useful when CSV was
    # generated on Windows and server paths changed after upload.
    basename = Path(raw).name
    if basename:
        matches = list(data_root.rglob(basename))
        matches = [m for m in matches if m.is_dir()]
        if len(matches) == 1:
            return matches[0]

    return candidates[0]


def canonical_activity(x: Any) -> Optional[str]:
    s = slug(x)
    return ACTIVITY_ALIASES.get(s)


def canonical_surface(x: Any) -> Optional[str]:
    raw = str(x).strip()
    if raw in SURFACE_ALIASES:
        return SURFACE_ALIASES[raw]
    return SURFACE_ALIASES.get(slug(raw))


def stage1_split(participant: str) -> Optional[str]:
    if participant in PARTICIPANT_STAGE1_TRAIN:
        return "train"
    if participant in PARTICIPANT_STAGE1_VAL:
        return "val"
    return None


def stage2_split(participant: str) -> Optional[str]:
    if participant in PARTICIPANT_STAGE2_TRAIN:
        return "train"
    if participant in PARTICIPANT_STAGE2_TEST:
        return "test"
    return None


def build_uniform_grid(t_min: float, t_max: float, step_ms: int = STEP_MS) -> np.ndarray:
    if t_max <= t_min:
        return np.array([], dtype=np.float32)
    n = int(np.floor((t_max - t_min) / step_ms)) + 1
    return (t_min + np.arange(n) * step_ms).astype(np.float32)


def interp_to_grid(
    t_src: List[float],
    y_src: List[List[float]],
    t_grid: np.ndarray,
    out_dim: int,
) -> np.ndarray:
    if len(t_grid) == 0:
        return np.zeros((0, out_dim), dtype=np.float32)

    if len(t_src) == 0:
        return np.zeros((len(t_grid), out_dim), dtype=np.float32)

    t = np.asarray(t_src, dtype=np.float32)
    y = np.asarray(y_src, dtype=np.float32)

    if y.ndim == 1:
        y = y[:, None]

    if y.shape[1] != out_dim:
        fixed = np.zeros((y.shape[0], out_dim), dtype=np.float32)
        c = min(out_dim, y.shape[1])
        fixed[:, :c] = y[:, :c]
        y = fixed

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    unique_t, unique_idx = np.unique(t, return_index=True)
    t = unique_t
    y = y[unique_idx]

    if len(t) == 1:
        return np.repeat(y[:1], len(t_grid), axis=0).astype(np.float32)

    out = np.zeros((len(t_grid), out_dim), dtype=np.float32)
    for c in range(out_dim):
        out[:, c] = np.interp(
            t_grid,
            t,
            y[:, c],
            left=y[0, c],
            right=y[-1, c],
        )

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# =========================
# Packet parsing
# =========================

def parse_packet(obj: Dict[str, Any]):
    """
    New-device packet format:
      ts
      IMU: list of a/g values
      tof: list of {"t": ..., "d": [...], "s": [...]}
      mag: x/y/z
    """
    ts = safe_float(obj.get("ts", 0.0), 0.0)

    t_imu, imu_values = [], []
    t_tof, tof_values = [], []
    t_mag, mag_values = [], []

    imu_list = obj.get("IMU", obj.get("imu", []))
    if isinstance(imu_list, list) and len(imu_list) > 0:
        n = len(imu_list)
        dt = 1000.0 / max(n, 1)

        for k, it in enumerate(imu_list):
            if not isinstance(it, dict):
                continue

            a = it.get("a", it.get("acc", {}))
            g = it.get("g", it.get("gyro", {}))
            if not isinstance(a, dict):
                a = {}
            if not isinstance(g, dict):
                g = {}

            row = [
                safe_float(a.get("x", 0.0)),
                safe_float(a.get("y", 0.0)),
                safe_float(a.get("z", 0.0)),
                safe_float(g.get("x", 0.0)),
                safe_float(g.get("y", 0.0)),
                safe_float(g.get("z", 0.0)),
            ]
            t_imu.append(ts + k * dt)
            imu_values.append(row)

    tof_list = obj.get("tof", obj.get("ToF", []))
    if isinstance(tof_list, list):
        for item in tof_list:
            if not isinstance(item, dict):
                continue

            raw_t = safe_float(item.get("t", 0.0), 0.0)
            d = item.get("d", None)
            s = item.get("s", None)

            if isinstance(d, list) and len(d) > 0:
                dist = safe_float(d[0], 0.0)
            else:
                dist = safe_float(item.get("d", 0.0), 0.0)

            if isinstance(s, list) and len(s) > 0:
                status = safe_float(s[0], 0.0)
            else:
                status = safe_float(item.get("s", 0.0), 0.0)

            if dist > 0:
                t_tof.append(ts + raw_t)
                tof_values.append([dist, status])

    m = obj.get("mag", {})
    if isinstance(m, dict) and len(m) > 0:
        row = [
            safe_float(m.get("x", 0.0)),
            safe_float(m.get("y", 0.0)),
            safe_float(m.get("z", 0.0)),
        ]
        t_mag.append(ts + 500.0)
        mag_values.append(row)

    return t_imu, imu_values, t_tof, tof_values, t_mag, mag_values


def parse_folder_series(folder: Path) -> Dict[str, np.ndarray]:
    pkt_files = sort_by_last_int(list(folder.glob("pkt_*.json")))

    t_imu_all, imu_all = [], []
    t_tof_all, tof_all = [], []
    t_mag_all, mag_all = [], []

    for p in pkt_files:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        t_imu, imu_values, t_tof, tof_values, t_mag, mag_values = parse_packet(obj)
        t_imu_all.extend(t_imu)
        imu_all.extend(imu_values)
        t_tof_all.extend(t_tof)
        tof_all.extend(tof_values)
        t_mag_all.extend(t_mag)
        mag_all.extend(mag_values)

    if len(t_imu_all) == 0:
        return {
            "t_grid": np.array([], dtype=np.float32),
            "imu": np.zeros((0, 6), dtype=np.float32),
            "tof": np.zeros((0, 2), dtype=np.float32),
            "mag": np.zeros((0, 3), dtype=np.float32),
        }

    t_min = float(np.min(t_imu_all))
    t_max = float(np.max(t_imu_all))
    t_grid = build_uniform_grid(t_min, t_max, STEP_MS)

    return {
        "t_grid": t_grid.astype(np.float32),
        "imu": interp_to_grid(t_imu_all, imu_all, t_grid, 6),
        "tof": interp_to_grid(t_tof_all, tof_all, t_grid, 2),
        "mag": interp_to_grid(t_mag_all, mag_all, t_grid, 3),
    }


# =========================
# Audio processing
# =========================

def load_folder_audio(folder: Path) -> Tuple[Optional[np.ndarray], Optional[int]]:
    wavs = sort_by_last_int(list(folder.glob("rec_*.wav")))
    if not wavs:
        return None, None

    times = []
    for p in wavs:
        t = parse_last_int_from_stem(p)
        if t is not None:
            times.append(t)

    if not times:
        return None, None

    base_ms = min(times)
    end_ms = max(times) + 1000
    total_samples = int((end_ms - base_ms) / 1000.0 * AUDIO_SR) + AUDIO_SR
    y_all = np.zeros(total_samples, dtype=np.float32)

    for p in wavs:
        t = parse_last_int_from_stem(p)
        if t is None:
            continue

        try:
            y, _ = librosa.load(str(p), sr=AUDIO_SR, mono=True)
        except Exception:
            continue

        start = int((t - base_ms) / 1000.0 * AUDIO_SR)
        end = min(start + len(y), len(y_all))
        if start < 0 or start >= len(y_all) or end <= start:
            continue
        y_all[start:end] = y[: end - start]

    return y_all, base_ms


def audio_segment_to_mel(
    y_all: Optional[np.ndarray],
    base_ms: Optional[int],
    start_ms: float,
) -> np.ndarray:
    out = np.zeros((AUDIO_STEPS, AUDIO_MELS, 1), dtype=np.float32)
    if y_all is None or base_ms is None:
        return out

    start_sample = int((start_ms - base_ms) / 1000.0 * AUDIO_SR)
    length = int(WIN_SEC * AUDIO_SR)

    if start_sample < 0:
        pad_left = -start_sample
        start_sample = 0
    else:
        pad_left = 0

    end_sample = start_sample + length - pad_left
    segment = y_all[start_sample : max(start_sample, end_sample)]

    if pad_left > 0:
        segment = np.pad(segment, (pad_left, 0))
    if len(segment) < length:
        segment = np.pad(segment, (0, length - len(segment)))
    else:
        segment = segment[:length]

    try:
        hop = max(1, len(segment) // AUDIO_STEPS)
        S = librosa.feature.melspectrogram(
            y=segment,
            sr=AUDIO_SR,
            n_mels=AUDIO_MELS,
            hop_length=hop,
            n_fft=512,
            power=2.0,
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        if S_db.shape[1] < AUDIO_STEPS:
            S_db = np.pad(S_db, ((0, 0), (0, AUDIO_STEPS - S_db.shape[1])))
        elif S_db.shape[1] > AUDIO_STEPS:
            S_db = S_db[:, :AUDIO_STEPS]

        mel = S_db.T.astype(np.float32)
        mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)

        if np.std(mel) > 1e-6:
            mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)

        return mel[..., None].astype(np.float32)
    except Exception:
        return out


# =========================
# RGB image processing
# =========================

def build_image_index(folder: Path, prefix: str) -> Tuple[List[int], List[Path]]:
    files = sort_by_last_int(list(folder.glob(f"{prefix}_*.jpg")))
    times = []
    valid_files = []

    for p in files:
        t = parse_last_int_from_stem(p)
        if t is not None:
            times.append(t)
            valid_files.append(p)

    if not times:
        return [], []

    order = np.argsort(times)
    times_sorted = [times[i] for i in order]
    files_sorted = [valid_files[i] for i in order]
    return times_sorted, files_sorted


def read_rgb_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path).convert("RGB") as im:
            im = im.resize(IMAGE_SIZE)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            return arr
    except Exception:
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)


def nearest_rgb_image(
    times: List[int],
    files: List[Path],
    target_ms: float,
    max_diff_ms: int = 800,
) -> np.ndarray:
    if not times or not files:
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

    idx = bisect.bisect_left(times, int(target_ms))
    candidates = []
    if idx < len(times):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

    best = min(candidates, key=lambda i: abs(times[i] - target_ms))
    if abs(times[best] - target_ms) > max_diff_ms:
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

    return read_rgb_image(files[best])


def extract_rgb_dual_image(
    img_index: Tuple[List[int], List[Path]],
    mid_index: Tuple[List[int], List[Path]],
    target_ms: float,
) -> np.ndarray:
    img_times, img_files = img_index
    mid_times, mid_files = mid_index
    img_rgb = nearest_rgb_image(img_times, img_files, target_ms)
    mid_rgb = nearest_rgb_image(mid_times, mid_files, target_ms)
    return np.concatenate([img_rgb, mid_rgb], axis=-1).astype(np.float32)


# =========================
# Window generation
# =========================

def generate_windows_for_folder(
    folder: Path,
    label_id: int,
    y_act_id: int,
    y_env_id: int,
    task_name: str,
    participant: str,
    label_activity: str,
    label_env: str,
) -> Dict[str, List[Any]]:
    series = parse_folder_series(folder)
    t_grid = series["t_grid"]
    imu = series["imu"]
    tof = series["tof"]
    mag = series["mag"]

    result = {
        "imu_win": [],
        "tof_win": [],
        "mag_win": [],
        "audio_win": [],
        "img_win": [],
        "mask": [],
        "y": [],
        "y_act": [],
        "y_env": [],
        "folder": [],
        "start_ms": [],
        "participant": [],
        "label_activity": [],
        "label_env": [],
        "task": [],
    }

    if len(t_grid) == 0 or len(imu) == 0:
        return result

    win_len = int(WIN_MS / STEP_MS)
    stride_len = int(STRIDE_MS / STEP_MS)
    if len(t_grid) < win_len:
        return result

    y_audio, audio_base_ms = load_folder_audio(folder)
    img_index = build_image_index(folder, "img")
    mid_index = build_image_index(folder, "mid")

    for start_idx in range(0, len(t_grid) - win_len + 1, stride_len):
        end_idx = start_idx + win_len

        w_imu = imu[start_idx:end_idx]
        w_tof = tof[start_idx:end_idx]
        w_mag = mag[start_idx:end_idx]

        start_ms = float(t_grid[start_idx])
        center_ms = float(t_grid[start_idx + win_len // 2])

        w_audio = audio_segment_to_mel(y_audio, audio_base_ms, start_ms)
        w_img = extract_rgb_dual_image(img_index, mid_index, center_ms)

        mask = np.array(
            [
                1.0 if np.any(np.abs(w_imu) > 1e-8) else 0.0,
                1.0 if np.any(np.abs(w_tof[..., 0]) > 1e-8) else 0.0,
                1.0 if np.any(np.abs(w_mag) > 1e-8) else 0.0,
                1.0 if np.any(np.abs(w_audio) > 1e-8) else 0.0,
                1.0 if np.any(np.abs(w_img) > 1e-8) else 0.0,
            ],
            dtype=np.float32,
        )

        result["imu_win"].append(w_imu.astype(np.float32))
        result["tof_win"].append(w_tof.astype(np.float32))
        result["mag_win"].append(w_mag.astype(np.float32))
        result["audio_win"].append(w_audio.astype(np.float32))
        result["img_win"].append(w_img.astype(np.float32))
        result["mask"].append(mask)
        result["y"].append(np.int64(label_id))
        result["y_act"].append(np.int64(y_act_id))
        result["y_env"].append(np.int64(y_env_id))
        result["folder"].append(str(folder))
        result["start_ms"].append(np.float32(start_ms))
        result["participant"].append(str(participant))
        result["label_activity"].append(str(label_activity))
        result["label_env"].append(str(label_env))
        result["task"].append(str(task_name))

    return result


def append_result(dst: Dict[str, List[Any]], src: Dict[str, List[Any]]) -> None:
    for k, v in src.items():
        if k not in dst:
            dst[k] = []
        dst[k].extend(v)


def empty_dataset() -> Dict[str, np.ndarray]:
    win_len = int(WIN_MS / STEP_MS)
    return {
        "imu_win": np.zeros((0, win_len, 6), dtype=np.float32),
        "tof_win": np.zeros((0, win_len, 2), dtype=np.float32),
        "mag_win": np.zeros((0, win_len, 3), dtype=np.float32),
        "audio_win": np.zeros((0, AUDIO_STEPS, AUDIO_MELS, 1), dtype=np.float32),
        "img_win": np.zeros((0, IMAGE_SIZE[0], IMAGE_SIZE[1], 6), dtype=np.float32),
        "mask": np.zeros((0, 5), dtype=np.float32),
        "y": np.zeros((0,), dtype=np.int64),
        "y_act": np.zeros((0,), dtype=np.int64),
        "y_env": np.zeros((0,), dtype=np.int64),
        "folder": np.array([], dtype=object),
        "start_ms": np.zeros((0,), dtype=np.float32),
        "participant": np.array([], dtype=object),
        "label_activity": np.array([], dtype=object),
        "label_env": np.array([], dtype=object),
        "task": np.array([], dtype=object),
    }


def finalize_dataset(data: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
    if not data or len(data.get("y", [])) == 0:
        return empty_dataset()

    out: Dict[str, np.ndarray] = {}
    for k in ["imu_win", "tof_win", "mag_win", "audio_win", "img_win", "mask"]:
        out[k] = np.stack(data[k], axis=0).astype(np.float32)

    for k in ["y", "y_act", "y_env"]:
        out[k] = np.asarray(data[k], dtype=np.int64)

    out["folder"] = np.asarray(data["folder"], dtype=object)
    out["start_ms"] = np.asarray(data["start_ms"], dtype=np.float32)
    out["participant"] = np.asarray(data["participant"], dtype=object)
    out["label_activity"] = np.asarray(data["label_activity"], dtype=object)
    out["label_env"] = np.asarray(data["label_env"], dtype=object)
    out["task"] = np.asarray(data["task"], dtype=object)
    return out


def save_npz(path: Path, data: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    log(f"[OK] saved {path}")


def make_card(
    version: str,
    task: str,
    classes: List[str],
    class_map: Dict[str, int],
    train_data: Dict[str, np.ndarray],
    eval_name: str,
    split_description: Dict[str, str],
) -> Dict[str, Any]:
    if task == "activity":
        activity_classes = classes
        env_classes = SURFACE_CLASSES
    elif task == "surface":
        activity_classes = ACTIVITY_CLASSES
        env_classes = classes
    else:
        raise ValueError(task)

    return {
        "version": version,
        "task": task,
        "classes": classes,
        "class_map": class_map,
        "label_key": "y",
        "activity_classes": activity_classes,
        "env_classes": env_classes,
        "activity_map": ACTIVITY_MAP,
        "env_map": SURFACE_MAP,
        "step_ms": STEP_MS,
        "win_sec": WIN_SEC,
        "stride_sec": STRIDE_SEC,
        "image_size": list(IMAGE_SIZE),
        "image_channels": 6,
        "image_description": "img RGB + mid RGB",
        "audio_description": f"log-mel spectrogram, {AUDIO_STEPS}x{AUDIO_MELS}, sr={AUDIO_SR}",
        "modalities": ALL_MODALITIES,
        "mask_order": MASK_ORDER,
        "eval_split_name": eval_name,
        "split_description": split_description,
        "shapes": {
            "imu_win": list(train_data["imu_win"].shape[1:]),
            "tof_win": list(train_data["tof_win"].shape[1:]),
            "mag_win": list(train_data["mag_win"].shape[1:]),
            "audio_win": list(train_data["audio_win"].shape[1:]),
            "img_win": list(train_data["img_win"].shape[1:]),
        },
        "notes": [
            "M14 final dataset.",
            "No stair-direction task is generated.",
            "Labels are taken from labels_all_m14.csv, not inferred from folder names.",
            "Surface task uses walking samples only.",
            "PVC is preserved as uppercase in label_env and class names.",
        ],
    }


def write_card(path: Path, card: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[OK] saved {path}")


def bincount_str(y: np.ndarray, classes: List[str]) -> str:
    if len(y) == 0:
        return "{}"
    counts = np.bincount(y.astype(int), minlength=len(classes))
    return json.dumps({name: int(counts[i]) for i, name in enumerate(classes)}, ensure_ascii=False)


def print_dataset_summary(name: str, d: Dict[str, np.ndarray], classes: List[str]) -> None:
    print(f"\n[{name}]")
    print("samples      =", len(d["y"]))
    print("class counts =", bincount_str(d["y"], classes))
    print("imu shape    =", d["imu_win"].shape)
    print("tof shape    =", d["tof_win"].shape)
    print("mag shape    =", d["mag_win"].shape)
    print("audio shape  =", d["audio_win"].shape)
    print("image shape  =", d["img_win"].shape)
    if len(d["mask"]) > 0:
        print("mask mean    =", d["mask"].mean(axis=0).round(4).tolist(), "order=", MASK_ORDER)
    if len(d["participant"]) > 0:
        print("participants =", dict(Counter(map(str, d["participant"]))))


# =========================
# Main dataset builder
# =========================

def build_all_datasets(df: pd.DataFrame, data_root: Path, folder_col: str, participant_col: str):
    buckets: Dict[str, Dict[str, Dict[str, List[Any]]]] = {
        "stage1": {
            "activity_train": {},
            "activity_val": {},
            "surface_train": {},
            "surface_val": {},
        },
        "stage2": {
            "activity_train": {},
            "activity_test": {},
            "surface_train": {},
            "surface_test": {},
        },
    }

    stats = defaultdict(int)
    folder_cache: Dict[Tuple[str, int, int, str, str, str, str], Dict[str, List[Any]]] = {}

    total = len(df)
    for idx, row in df.iterrows():
        participant = str(row[participant_col]).strip()
        act = canonical_activity(row["label_activity"])
        env = canonical_surface(row["label_env"])

        if act is None:
            stats["skipped_unknown_activity_or_stairs"] += 1
            continue
        if env is None:
            stats["skipped_unknown_surface"] += 1
            continue

        folder = resolve_folder(data_root, row[folder_col])
        if not folder.exists():
            stats["skipped_missing_folder"] += 1
            continue

        y_act_id = ACTIVITY_MAP[act]
        y_env_id = SURFACE_MAP[env]

        # Generate activity windows once.
        activity_key = (str(folder), y_act_id, y_env_id, "activity", participant, act, env)
        if activity_key not in folder_cache:
            folder_cache[activity_key] = generate_windows_for_folder(
                folder=folder,
                label_id=y_act_id,
                y_act_id=y_act_id,
                y_env_id=y_env_id,
                task_name="activity",
                participant=participant,
                label_activity=act,
                label_env=env,
            )
        activity_res = folder_cache[activity_key]
        if len(activity_res["y"]) == 0:
            stats["skipped_no_windows_activity"] += 1
            continue

        s1 = stage1_split(participant)
        s2 = stage2_split(participant)

        if s1 == "train":
            append_result(buckets["stage1"]["activity_train"], activity_res)
        elif s1 == "val":
            append_result(buckets["stage1"]["activity_val"], activity_res)
        else:
            stats["stage1_activity_not_used"] += 1

        if s2 == "train":
            append_result(buckets["stage2"]["activity_train"], activity_res)
        elif s2 == "test":
            append_result(buckets["stage2"]["activity_test"], activity_res)
        else:
            stats["stage2_activity_not_used"] += 1

        # Surface task: walking only.
        if act == "walk":
            surface_key = (str(folder), y_act_id, y_env_id, "surface", participant, act, env)
            if surface_key not in folder_cache:
                folder_cache[surface_key] = generate_windows_for_folder(
                    folder=folder,
                    label_id=y_env_id,
                    y_act_id=y_act_id,
                    y_env_id=y_env_id,
                    task_name="surface",
                    participant=participant,
                    label_activity=act,
                    label_env=env,
                )
            surface_res = folder_cache[surface_key]
            if len(surface_res["y"]) == 0:
                stats["skipped_no_windows_surface"] += 1
            else:
                if s1 == "train":
                    append_result(buckets["stage1"]["surface_train"], surface_res)
                elif s1 == "val":
                    append_result(buckets["stage1"]["surface_val"], surface_res)
                else:
                    stats["stage1_surface_not_used"] += 1

                if s2 == "train":
                    append_result(buckets["stage2"]["surface_train"], surface_res)
                elif s2 == "test":
                    append_result(buckets["stage2"]["surface_test"], surface_res)
                else:
                    stats["stage2_surface_not_used"] += 1
        else:
            stats["surface_skipped_non_walking"] += 1

        if (idx + 1) % 50 == 0:
            log(f"[INFO] processed folders: {idx + 1}/{total}")

    finalized = {
        stage: {name: finalize_dataset(data) for name, data in split_dict.items()}
        for stage, split_dict in buckets.items()
    }
    return finalized, stats


def save_stage_outputs(out_root: Path, stage_name: str, data: Dict[str, Dict[str, np.ndarray]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    if stage_name == "stage1":
        eval_name = "val"
        split_description = {
            "train": "P1and2",
            "val": "P3",
        }
        act_eval_key = "activity_val"
        surf_eval_key = "surface_val"
        eval_npz_name = "dataset_val.npz"
    elif stage_name == "stage2":
        eval_name = "test"
        split_description = {
            "train": "P1and2 + P3",
            "test": "P4",
        }
        act_eval_key = "activity_test"
        surf_eval_key = "surface_test"
        eval_npz_name = "dataset_test.npz"
    else:
        raise ValueError(stage_name)

    # Activity
    act_dir = out_root / "activity"
    save_npz(act_dir / "dataset_train.npz", data["activity_train"])
    save_npz(act_dir / eval_npz_name, data[act_eval_key])
    act_card = make_card(
        version=f"M14_RGB64_{stage_name}",
        task="activity",
        classes=ACTIVITY_CLASSES,
        class_map=ACTIVITY_MAP,
        train_data=data["activity_train"],
        eval_name=eval_name,
        split_description=split_description,
    )
    write_card(act_dir / "dataset_card.json", act_card)

    # Surface
    surf_dir = out_root / "surface"
    save_npz(surf_dir / "dataset_train.npz", data["surface_train"])
    save_npz(surf_dir / eval_npz_name, data[surf_eval_key])
    surf_card = make_card(
        version=f"M14_RGB64_{stage_name}",
        task="surface",
        classes=SURFACE_CLASSES,
        class_map=SURFACE_MAP,
        train_data=data["surface_train"],
        eval_name=eval_name,
        split_description=split_description,
    )
    write_card(surf_dir / "dataset_card.json", surf_card)

    print_dataset_summary(f"{stage_name} activity train", data["activity_train"], ACTIVITY_CLASSES)
    print_dataset_summary(f"{stage_name} activity {eval_name}", data[act_eval_key], ACTIVITY_CLASSES)
    print_dataset_summary(f"{stage_name} surface train", data["surface_train"], SURFACE_CLASSES)
    print_dataset_summary(f"{stage_name} surface {eval_name}", data[surf_eval_key], SURFACE_CLASSES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=str, help="Path to labels_all_m14.csv")
    parser.add_argument("--data_root", required=True, type=str, help="Root folder containing P1and2/P3/P4 data")
    parser.add_argument("--out_stage1", required=True, type=str, help="Output root for Stage 1 datasets")
    parser.add_argument("--out_stage2", required=True, type=str, help="Output root for Stage 2 datasets")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: process first N CSV rows")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    data_root = Path(args.data_root)
    out_stage1 = Path(args.out_stage1)
    out_stage2 = Path(args.out_stage2)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = {"label_activity", "label_env"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Existing columns: {df.columns.tolist()}")

    folder_col = find_folder_column(df)
    participant_col = find_participant_column(df)

    if args.limit and args.limit > 0:
        df = df.head(int(args.limit)).copy()
        log(f"[WARN] Debug limit enabled: only processing first {len(df)} rows")

    print("[INFO] CSV:", csv_path)
    print("[INFO] data_root:", data_root)
    print("[INFO] folder_col:", folder_col)
    print("[INFO] participant_col:", participant_col)
    print("[INFO] rows:", len(df))
    print("[INFO] participant counts:", dict(Counter(map(str, df[participant_col]))))
    print("[INFO] label_activity counts:", dict(Counter(map(str, df["label_activity"]))))
    print("[INFO] label_env counts:", dict(Counter(map(str, df["label_env"]))))

    finalized, stats = build_all_datasets(df, data_root, folder_col, participant_col)

    save_stage_outputs(out_stage1, "stage1", finalized["stage1"])
    save_stage_outputs(out_stage2, "stage2", finalized["stage2"])

    stats_path_1 = out_stage1 / "build_stats.json"
    stats_path_2 = out_stage2 / "build_stats.json"
    stats_dict = {k: int(v) for k, v in sorted(stats.items())}
    stats_path_1.write_text(json.dumps(stats_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    stats_path_2.write_text(json.dumps(stats_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[BUILD STATS]")
    print(json.dumps(stats_dict, indent=2, ensure_ascii=False))
    print("\n[OK] M14 RGB64 dataset preparation finished.")


if __name__ == "__main__":
    main()
