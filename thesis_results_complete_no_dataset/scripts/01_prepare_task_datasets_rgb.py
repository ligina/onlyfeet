# scripts/01_prepare_task_datasets_rgb.py
# -*- coding: utf-8 -*-
"""
OnlyFeet M13 RGB Dataset Preparation

This script rebuilds task datasets directly from raw folders and labels_all.csv.

Main changes compared with grayscale version:
  - Image input becomes RGB:
      img_rgb + mid_rgb => (H, W, 6)
  - Default image size is 96x96.
  - Regular task excludes stairs.
  - Stairs task excludes environment label.

Outputs:
  datasets_m13_rgb/
    regular/
      dataset_train.npz
      dataset_val.npz
      dataset_card.json

    stairs/
      dataset_train.npz
      dataset_val.npz
      dataset_card.json

Regular task:
  activity: walk / standing / sitting
  environment: asphalt / pvc / sand / gravel / grass

Stairs task:
  activity: stairs_up / stairs_down
  environment ignored
"""

import argparse
import bisect
import json
import math
import re
import warnings
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
# Config
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

REGULAR_ACTIVITIES = ["walk", "standing", "sitting"]
STAIR_ACTIVITIES = ["stairs_up", "stairs_down"]
ENV_CLASSES = ["asphalt", "pvc", "sand", "gravel", "grass"]

ALL_MODALITIES = ["imu", "tof", "mag", "audio", "image"]
MASK_ORDER = ["imu", "tof", "mag", "audio", "image"]


# =========================
# Utilities
# =========================

def log(msg: str):
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
    ]

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Cannot find folder path column. Existing columns: {df.columns.tolist()}"
    )


def resolve_folder(data_root: Path, value: Any) -> Path:
    """
    Supports:
      1. absolute path
      2. path relative to data_root
      3. path relative to project root
      4. path starting with data/
    """
    raw = str(value).strip().replace("\\", "/")
    p = Path(raw)

    if p.is_absolute() and p.exists():
        return p

    project_root = data_root.parent

    candidates = []

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

    return candidates[0]


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

    # IMU
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

    # ToF
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

    # Mag
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

    imu_grid = interp_to_grid(t_imu_all, imu_all, t_grid, 6)
    tof_grid = interp_to_grid(t_tof_all, tof_all, t_grid, 2)
    mag_grid = interp_to_grid(t_mag_all, mag_all, t_grid, 3)

    return {
        "t_grid": t_grid.astype(np.float32),
        "imu": imu_grid.astype(np.float32),
        "tof": tof_grid.astype(np.float32),
        "mag": mag_grid.astype(np.float32),
    }


# =========================
# Audio
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

        y_all[start:end] = y[:end - start]

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
    segment = y_all[start_sample:max(start_sample, end_sample)]

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
# RGB image
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
    act_id: int,
    env_id: int,
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
        "y_act": [],
        "y_env": [],
        "folder": [],
        "start_ms": [],
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
        result["y_act"].append(np.int64(act_id))
        result["y_env"].append(np.int64(env_id))
        result["folder"].append(str(folder))
        result["start_ms"].append(np.float32(start_ms))

    return result


def append_result(dst: Dict[str, List[Any]], src: Dict[str, List[Any]]):
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
        "y_act": np.zeros((0,), dtype=np.int64),
        "y_env": np.zeros((0,), dtype=np.int64),
        "folder": np.array([], dtype=object),
        "start_ms": np.zeros((0,), dtype=np.float32),
    }


def finalize_dataset(data: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
    if not data or len(data.get("y_act", [])) == 0:
        return empty_dataset()

    out = {}

    for k in ["imu_win", "tof_win", "mag_win", "audio_win", "img_win", "mask"]:
        out[k] = np.stack(data[k], axis=0).astype(np.float32)

    out["y_act"] = np.asarray(data["y_act"], dtype=np.int64)
    out["y_env"] = np.asarray(data["y_env"], dtype=np.int64)
    out["folder"] = np.asarray(data["folder"], dtype=object)
    out["start_ms"] = np.asarray(data["start_ms"], dtype=np.float32)

    return out


def save_npz(path: Path, data: Dict[str, np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f"[OK] saved {path}")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True, type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)

    args = parser.parse_args()

    csv_path = Path(args.csv)
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = {"split", "label_activity", "label_env"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Existing: {df.columns.tolist()}")

    folder_col = find_folder_column(df)

    regular_act_map = {name: i for i, name in enumerate(REGULAR_ACTIVITIES)}
    stair_act_map = {name: i for i, name in enumerate(STAIR_ACTIVITIES)}
    env_map = {name: i for i, name in enumerate(ENV_CLASSES)}

    regular_train: Dict[str, List[Any]] = {}
    regular_val: Dict[str, List[Any]] = {}
    stairs_train: Dict[str, List[Any]] = {}
    stairs_val: Dict[str, List[Any]] = {}

    skipped_missing_folder = 0
    skipped_unknown = 0
    skipped_no_windows = 0

    total = len(df)

    for idx, row in df.iterrows():
        split = slug(row["split"])
        act_name = slug(row["label_activity"])
        env_name = slug(row["label_env"])

        folder = resolve_folder(data_root, row[folder_col])

        if not folder.exists():
            skipped_missing_folder += 1
            continue

        # Regular branch
        if act_name in regular_act_map:
            if env_name not in env_map:
                skipped_unknown += 1
                continue

            act_id = regular_act_map[act_name]
            env_id = env_map[env_name]

            res = generate_windows_for_folder(folder, act_id, env_id)

            if len(res["y_act"]) == 0:
                skipped_no_windows += 1
                continue

            if split == "train":
                append_result(regular_train, res)
            elif split in {"val", "valid", "validation"}:
                append_result(regular_val, res)
            else:
                skipped_unknown += 1

        # Stairs branch
        elif act_name in stair_act_map:
            act_id = stair_act_map[act_name]
            env_id = -1

            res = generate_windows_for_folder(folder, act_id, env_id)

            if len(res["y_act"]) == 0:
                skipped_no_windows += 1
                continue

            if split == "train":
                append_result(stairs_train, res)
            elif split in {"val", "valid", "validation"}:
                append_result(stairs_val, res)
            else:
                skipped_unknown += 1

        else:
            skipped_unknown += 1

        if (idx + 1) % 100 == 0:
            log(f"[INFO] processed folders: {idx + 1}/{total}")

    regular_train_npz = finalize_dataset(regular_train)
    regular_val_npz = finalize_dataset(regular_val)
    stairs_train_npz = finalize_dataset(stairs_train)
    stairs_val_npz = finalize_dataset(stairs_val)

    save_npz(out_dir / "regular" / "dataset_train.npz", regular_train_npz)
    save_npz(out_dir / "regular" / "dataset_val.npz", regular_val_npz)
    save_npz(out_dir / "stairs" / "dataset_train.npz", stairs_train_npz)
    save_npz(out_dir / "stairs" / "dataset_val.npz", stairs_val_npz)

    regular_card = {
        "version": "M13_RGB",
        "task": "regular_activity_and_surface",
        "step_ms": STEP_MS,
        "win_sec": WIN_SEC,
        "stride_sec": STRIDE_SEC,
        "image_size": list(IMAGE_SIZE),
        "image_channels": 6,
        "image_description": "img RGB + mid RGB",
        "modalities": ALL_MODALITIES,
        "mask_order": MASK_ORDER,
        "activity_classes": REGULAR_ACTIVITIES,
        "env_classes": ENV_CLASSES,
        "activity_map": regular_act_map,
        "env_map": env_map,
        "static_indices": [
            regular_act_map["standing"],
            regular_act_map["sitting"],
        ],
        "shapes": {
            "imu_win": list(regular_train_npz["imu_win"].shape[1:]),
            "tof_win": list(regular_train_npz["tof_win"].shape[1:]),
            "mag_win": list(regular_train_npz["mag_win"].shape[1:]),
            "audio_win": list(regular_train_npz["audio_win"].shape[1:]),
            "img_win": list(regular_train_npz["img_win"].shape[1:]),
        },
        "notes": [
            "RGB version.",
            "Regular dataset excludes stairs_up and stairs_down.",
            "Surface recognition is only trained on regular samples.",
        ],
    }

    stairs_card = {
        "version": "M13_RGB",
        "task": "stair_direction",
        "step_ms": STEP_MS,
        "win_sec": WIN_SEC,
        "stride_sec": STRIDE_SEC,
        "image_size": list(IMAGE_SIZE),
        "image_channels": 6,
        "image_description": "img RGB + mid RGB",
        "modalities": ALL_MODALITIES,
        "mask_order": MASK_ORDER,
        "activity_classes": STAIR_ACTIVITIES,
        "env_classes": [],
        "activity_map": stair_act_map,
        "env_map": {},
        "static_indices": [],
        "shapes": {
            "imu_win": list(stairs_train_npz["imu_win"].shape[1:]),
            "tof_win": list(stairs_train_npz["tof_win"].shape[1:]),
            "mag_win": list(stairs_train_npz["mag_win"].shape[1:]),
            "audio_win": list(stairs_train_npz["audio_win"].shape[1:]),
            "img_win": list(stairs_train_npz["img_win"].shape[1:]),
        },
        "notes": [
            "RGB version.",
            "Stairs dataset only contains stairs_up and stairs_down.",
            "Environment label is ignored for stair samples.",
        ],
    }

    (out_dir / "regular").mkdir(parents=True, exist_ok=True)
    (out_dir / "stairs").mkdir(parents=True, exist_ok=True)

    (out_dir / "regular" / "dataset_card.json").write_text(
        json.dumps(regular_card, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    (out_dir / "stairs" / "dataset_card.json").write_text(
        json.dumps(stairs_card, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    def print_summary(name: str, d: Dict[str, np.ndarray], act_classes: List[str], env_classes: List[str]):
        print(f"\n[{name}]")
        print("samples =", len(d["y_act"]))
        print("img shape =", d["img_win"].shape)
        print("act bincount =", np.bincount(d["y_act"], minlength=len(act_classes)).tolist())

        if env_classes and np.any(d["y_env"] >= 0):
            print("env bincount =", np.bincount(d["y_env"], minlength=len(env_classes)).tolist())

        if len(d["mask"]) > 0:
            print("mask mean =", d["mask"].mean(axis=0).round(4).tolist())

    print_summary("regular train", regular_train_npz, REGULAR_ACTIVITIES, ENV_CLASSES)
    print_summary("regular val", regular_val_npz, REGULAR_ACTIVITIES, ENV_CLASSES)
    print_summary("stairs train", stairs_train_npz, STAIR_ACTIVITIES, [])
    print_summary("stairs val", stairs_val_npz, STAIR_ACTIVITIES, [])

    print("\n[INFO] skipped_missing_folder =", skipped_missing_folder)
    print("[INFO] skipped_unknown        =", skipped_unknown)
    print("[INFO] skipped_no_windows     =", skipped_no_windows)
    print("[OK] RGB dataset preparation finished.")


if __name__ == "__main__":
    main()