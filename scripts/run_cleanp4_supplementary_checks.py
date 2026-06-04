from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "onlyfeet" / "analysis_outputs" / "cleanp4_supplementary_checks"


TASKS = {
    "activity": {
        "dataset_dir": ROOT / "onlyfeet" / "datasets_m14_rgb64_stage2" / "activity",
        "test_npz": ROOT / "onlyfeet" / "datasets_m14_rgb64_stage2" / "activity" / "dataset_test.npz",
        "card": ROOT / "onlyfeet" / "datasets_m14_rgb64_stage2" / "activity" / "dataset_card.json",
        "prediction_csv": ROOT
        / "onlyfeet"
        / "models"
        / "clean_p4_final"
        / "stage2_activity_imu_single_seed42_cleanp4"
        / "eval_predictions.csv",
        "metrics_json": ROOT
        / "onlyfeet"
        / "models"
        / "clean_p4_final"
        / "stage2_activity_imu_single_seed42_cleanp4"
        / "eval_metrics.json",
        "label_key": "y",
        "label_name_key": "label_activity",
    },
    "surface": {
        "dataset_dir": ROOT / "onlyfeet" / "datasets_m14_rgb64_stage2" / "surface",
        "test_npz": ROOT / "onlyfeet" / "datasets_m14_rgb64_stage2" / "surface" / "dataset_test.npz",
        "card": ROOT / "onlyfeet" / "datasets_m14_rgb64_stage2" / "surface" / "dataset_card.json",
        "prediction_csv": ROOT
        / "onlyfeet"
        / "models"
        / "clean_p4_final"
        / "stage2_surface_image_audio_concat_seed42_cleanp4"
        / "eval_predictions.csv",
        "metrics_json": ROOT
        / "onlyfeet"
        / "models"
        / "clean_p4_final"
        / "stage2_surface_image_audio_concat_seed42_cleanp4"
        / "eval_metrics.json",
        "label_key": "y",
        "label_name_key": "label_env",
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def majority_label(values: list[int]) -> tuple[int, bool]:
    counts = Counter(values)
    max_count = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == max_count)
    return winners[0], len(winners) > 1


def accuracy_score_local(y_true: list[int], y_pred: list[int]) -> float:
    if not y_true:
        return 0.0
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


def macro_f1_zero_division_0(y_true: list[int], y_pred: list[int], labels: list[int]) -> float:
    scores = []
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        scores.append(f1)
    return sum(scores) / len(scores) if scores else 0.0


def metric_row(task: str, y_true: list[int], y_pred: list[int], class_names: list[str], extra: dict) -> dict:
    labels = list(range(len(class_names)))
    row = {
        "task": task,
        "n": len(y_true),
        "accuracy": accuracy_score_local(y_true, y_pred),
        "macro_f1": macro_f1_zero_division_0(y_true, y_pred, labels),
        "class_names": "|".join(class_names),
    }
    row.update(extra)
    return row


def analyze_task(task: str, cfg: dict) -> dict:
    card = read_json(cfg["card"])
    class_names = card["classes"]
    metrics = read_json(cfg["metrics_json"])
    prediction_rows = read_predictions(cfg["prediction_csv"])

    with np.load(cfg["test_npz"], allow_pickle=True) as npz:
        folders = npz["folder"]
        true_labels = npz[cfg["label_key"]].astype(int)
        label_names = npz[cfg["label_name_key"]]

    pred_by_idx = {int(row["idx"]): int(row["y_pred"]) for row in prediction_rows}
    true_by_idx = {int(row["idx"]): int(row["y_true"]) for row in prediction_rows}
    missing_indices = [i for i in range(len(true_labels)) if i not in pred_by_idx]

    if missing_indices:
        raise RuntimeError(f"{task}: missing predictions for {len(missing_indices)} windows")

    y_true_window = [int(true_labels[i]) for i in range(len(true_labels))]
    y_pred_window = [pred_by_idx[i] for i in range(len(true_labels))]
    csv_true_mismatch = [
        i for i in range(len(true_labels)) if true_by_idx.get(i) is not None and true_by_idx[i] != int(true_labels[i])
    ]

    grouped: dict[str, dict[str, list]] = defaultdict(lambda: {"true": [], "pred": [], "label_names": []})
    for i, folder in enumerate(folders):
        folder_str = str(folder)
        grouped[folder_str]["true"].append(y_true_window[i])
        grouped[folder_str]["pred"].append(y_pred_window[i])
        grouped[folder_str]["label_names"].append(str(label_names[i]))

    per_folder = []
    folder_y_true = []
    folder_y_pred = []
    inconsistent_folders = []
    tie_folders = []

    for folder, vals in sorted(grouped.items()):
        true_counts = Counter(vals["true"])
        folder_true, true_tie = majority_label(vals["true"])
        if len(true_counts) > 1 or true_tie:
            inconsistent_folders.append(folder)
        pred, pred_tie = majority_label(vals["pred"])
        if pred_tie:
            tie_folders.append(folder)
        folder_y_true.append(folder_true)
        folder_y_pred.append(pred)
        per_folder.append(
            {
                "task": task,
                "folder": folder,
                "n_windows": len(vals["true"]),
                "y_true": folder_true,
                "y_pred": pred,
                "y_true_name": class_names[folder_true],
                "y_pred_name": class_names[pred],
                "correct": int(folder_true == pred),
                "window_accuracy_in_folder": accuracy_score_local(vals["true"], vals["pred"]),
                "pred_counts": dict(sorted(Counter(vals["pred"]).items())),
                "true_counts": dict(sorted(true_counts.items())),
                "prediction_tie": pred_tie,
            }
        )

    folder_summary = metric_row(
        task,
        folder_y_true,
        folder_y_pred,
        class_names,
        {
            "level": "folder_majority_vote",
            "n_windows": len(y_true_window),
            "n_folders": len(folder_y_true),
            "prediction_source": str(cfg["prediction_csv"]),
            "dataset_npz": str(cfg["test_npz"]),
            "tie_breaking": "smallest class index",
            "n_prediction_tie_folders": len(tie_folders),
            "n_inconsistent_true_label_folders": len(inconsistent_folders),
        },
    )

    majority_class, _ = majority_label(y_true_window)
    y_pred_majority = [majority_class] * len(y_true_window)
    window_baseline = metric_row(
        task,
        y_true_window,
        y_pred_majority,
        class_names,
        {
            "level": "window_majority_class_baseline",
            "majority_class": majority_class,
            "majority_class_name": class_names[majority_class],
            "n_windows": len(y_true_window),
            "n_folders": len(folder_y_true),
            "dataset_npz": str(cfg["test_npz"]),
        },
    )

    folder_majority_class, _ = majority_label(folder_y_true)
    folder_baseline = metric_row(
        task,
        folder_y_true,
        [folder_majority_class] * len(folder_y_true),
        class_names,
        {
            "level": "folder_majority_class_baseline",
            "majority_class": folder_majority_class,
            "majority_class_name": class_names[folder_majority_class],
            "n_windows": len(y_true_window),
            "n_folders": len(folder_y_true),
            "dataset_npz": str(cfg["test_npz"]),
        },
    )

    return {
        "task": task,
        "class_names": class_names,
        "dataset_dir": str(cfg["dataset_dir"]),
        "test_npz": str(cfg["test_npz"]),
        "dataset_card": str(cfg["card"]),
        "prediction_source": str(cfg["prediction_csv"]),
        "metrics_source": str(cfg["metrics_json"]),
        "window_final_model_metrics": metrics,
        "n_windows": len(y_true_window),
        "n_folders": len(folder_y_true),
        "folder_summary": folder_summary,
        "baseline_rows": [window_baseline, folder_baseline],
        "per_folder": per_folder,
        "warnings": {
            "csv_true_label_mismatch_indices": csv_true_mismatch,
            "inconsistent_true_label_folders": inconsistent_folders,
            "prediction_tie_folders": tie_folders,
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(results: dict[str, dict]) -> None:
    lines = [
        "# Clean-P4 Supplementary Checks",
        "",
        "These outputs are supplementary diagnostics only. They do not replace the main Clean-P4 window-level evaluation and do not constitute new training experiments.",
        "",
        "## Data Sources",
        "",
    ]
    for task, result in results.items():
        lines.extend(
            [
                f"### {task.title()}",
                f"- Dataset path: `{result['test_npz']}`",
                f"- Dataset card: `{result['dataset_card']}`",
                f"- Prediction source: `{result['prediction_source']}`",
                f"- Existing final model metrics source: `{result['metrics_source']}`",
                f"- Class names: {', '.join(result['class_names'])}",
                f"- Windows: {result['n_windows']}",
                f"- Unique folders: {result['n_folders']}",
                "",
            ]
        )

    lines.extend(["## Results", ""])
    for task, result in results.items():
        final_metrics = result["window_final_model_metrics"]
        folder = result["folder_summary"]
        baselines = {row["level"]: row for row in result["baseline_rows"]}
        wb = baselines["window_majority_class_baseline"]
        fb = baselines["folder_majority_class_baseline"]
        lines.extend(
            [
                f"### {task.title()}",
                f"- Existing Clean-P4 window-level final model accuracy: {pct(final_metrics['accuracy'])}",
                f"- Existing Clean-P4 window-level final model macro-F1: {pct(final_metrics['macro_f1'])}",
                f"- Folder-level majority-vote accuracy: {pct(folder['accuracy'])}",
                f"- Folder-level majority-vote macro-F1: {pct(folder['macro_f1'])}",
                f"- Folders used: {folder['n_folders']}",
                f"- Window-level majority-class baseline: majority class `{wb['majority_class_name']}`, accuracy {pct(wb['accuracy'])}, macro-F1 {pct(wb['macro_f1'])}",
                f"- Folder-level majority-class baseline: majority class `{fb['majority_class_name']}`, accuracy {pct(fb['accuracy'])}, macro-F1 {pct(fb['macro_f1'])}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Folder-level majority vote reduces the influence of multiple correlated windows from the same recording folder by aggregating predictions before computing metrics.",
            "- The majority-class sanity baseline checks whether performance can be explained by class imbalance alone.",
            "- These checks support interpretation of the Clean-P4 evaluation, but they do not remove all limitations of the controlled collection protocol.",
            "- These diagnostics do not replace the main Clean-P4 window-level evaluation.",
            "",
            "## Warnings and Uncertainties",
            "",
            "- Existing Clean-P4 prediction files were found and used; no inference or training was run.",
            "- Folder-level prediction ties are resolved by selecting the smallest class index.",
        ]
    )
    for task, result in results.items():
        warnings = result["warnings"]
        if warnings["csv_true_label_mismatch_indices"]:
            lines.append(f"- {task}: CSV true labels differed from NPZ labels at {len(warnings['csv_true_label_mismatch_indices'])} indices.")
        if warnings["inconsistent_true_label_folders"]:
            lines.append(f"- {task}: {len(warnings['inconsistent_true_label_folders'])} folders had inconsistent true labels.")
        if warnings["prediction_tie_folders"]:
            lines.append(f"- {task}: {len(warnings['prediction_tie_folders'])} folders had prediction ties.")
        if not any(warnings.values()):
            lines.append(f"- {task}: no true-label inconsistencies, prediction ties, or CSV/NPZ label mismatches were detected.")
    lines.append("")
    (OUT_DIR / "cleanp4_supplementary_checks.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {task: analyze_task(task, cfg) for task, cfg in TASKS.items()}

    folder_rows = [result["folder_summary"] for result in results.values()]
    baseline_rows = []
    for result in results.values():
        baseline_rows.extend(result["baseline_rows"])

    write_csv(OUT_DIR / "cleanp4_folder_majority_vote.csv", folder_rows)
    write_csv(OUT_DIR / "cleanp4_majority_class_baseline.csv", baseline_rows)

    (OUT_DIR / "cleanp4_folder_majority_vote.json").write_text(
        json.dumps({task: results[task]["folder_summary"] | {"per_folder": results[task]["per_folder"]} for task in results}, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "cleanp4_majority_class_baseline.json").write_text(
        json.dumps({task: results[task]["baseline_rows"] for task in results}, indent=2),
        encoding="utf-8",
    )
    write_report(results)

    print(json.dumps({task: {"folder": results[task]["folder_summary"], "baselines": results[task]["baseline_rows"]} for task in results}, indent=2))


if __name__ == "__main__":
    main()
