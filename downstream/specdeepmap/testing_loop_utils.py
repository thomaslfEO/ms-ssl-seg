"""Shared helpers for batch checkpoint evaluation loops."""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Callable

import numpy as np

WEIGHT_DISPLAY = {
    "imagenet": "ImageNet",
    "none": "None (random)",
    "micasense_sr_swin_s3_tiny": "SSL-MoCo-Swin",
    "mae_vit_small": "MAE-ViT-Small",
    "mae_vit_base": "MAE-ViT-Base",
    "micasense_sr_swin_s3_tiny_fastsiam": "SSL-FastSiam",
}

# train_loop.py layout:  imagenet_frozen
CONFIG_DIR_PATTERN = re.compile(r"^(.+)_(frozen|unfrozen)$")

# train_loop_stats.py layout:  imagenet_frozen_run3
RUN_DIR_PATTERN = re.compile(r"^(.+)_(frozen|unfrozen)_run(\d+)$")


def find_best_checkpoint(run_dir: str) -> str | None:
    """Return the best *.ckpt in a directory (highest val_iou in filename)."""
    if not os.path.isdir(run_dir):
        return None

    ckpts = [f for f in os.listdir(run_dir) if f.endswith(".ckpt")]
    if not ckpts:
        return None

    def _score(fname: str) -> float:
        match = re.search(r"val_iou[_=]([\d.]+)", fname)
        return float(match.group(1)) if match else -1.0

    best = max(ckpts, key=_score)
    return os.path.join(run_dir, best)


def read_mean_metrics_from_csv(csv_path: str) -> dict | None:
    """Read the Mean row from a test_results.csv file."""
    if not os.path.isfile(csv_path):
        return None
    try:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if str(row.get("Class", "")).strip().lower() == "mean":
                    return {
                        "iou": float(row["IoU"]),
                        "accuracy": float(row["Accuracy"]),
                        "recall": float(row["Recall"]),
                        "precision": float(row["Precision"]),
                        "f1": float(row["F1_Score"]),
                    }
    except Exception as exc:
        print(f"  WARNING: could not read {csv_path}: {exc}")
    return None


def run_test_on_checkpoint(
    process_fn: Callable,
    *,
    ckpt_path: str,
    test_csv: str,
    output_csv: str,
    acc_device: int,
    feedback,
) -> dict | None:
    """Run the tester and return mean metrics, or None on failure."""
    try:
        process_fn(
            csv_file=test_csv,
            model_checkpoint=ckpt_path,
            acc_device=acc_device,
            csv_output_path=output_csv,
            export_folder=None,
            no_data_label_mask=True,
            feedback=feedback,
        )
        return read_mean_metrics_from_csv(output_csv)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"  ERROR testing {ckpt_path}: {exc}")
        import traceback

        traceback.print_exc()
        return None


def resolve_test_csv(
    splits_root: str,
    split_name: str,
    global_test_csv: str | None,
) -> str | None:
    """Pick test CSV for a split (test_files.csv, then validation_files.csv)."""
    if global_test_csv:
        return global_test_csv if os.path.isfile(global_test_csv) else None

    for name in ("test_files.csv", "validation_files.csv"):
        candidate = os.path.join(splits_root, split_name, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def discover_config_dirs(checkpoints_root: str) -> list[tuple[str, str, str, str]]:
    """
    Discover experiment directories from train_loop.py.

    Returns list of (split_name, weight_name, freeze_str, full_path).
    """
    entries: list[tuple[str, str, str, str]] = []

    if not os.path.isdir(checkpoints_root):
        return entries

    for split_name in sorted(os.listdir(checkpoints_root)):
        split_dir = os.path.join(checkpoints_root, split_name)
        if not os.path.isdir(split_dir):
            continue

        for dirname in sorted(os.listdir(split_dir)):
            match = CONFIG_DIR_PATTERN.match(dirname)
            if not match:
                continue
            entries.append(
                (split_name, match.group(1), match.group(2), os.path.join(split_dir, dirname))
            )
    return entries


def write_results_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_results_json(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def print_results_table(rows: list[dict]) -> None:
    if not rows:
        return

    print("\n" + "=" * 90)
    print(
        f"{'Split':<12} {'Weights':<22} {'Freeze':<10} "
        f"{'mIoU':>8} {'Acc':>8} {'F1':>8}"
    )
    print("-" * 90)
    for row in rows:
        weight = WEIGHT_DISPLAY.get(row["weight_name"], row["weight_name"])
        print(
            f"{row['split']:<12} "
            f"{weight:<22} "
            f"{row['freeze']:<10} "
            f"{row.get('iou', float('nan')):>8.4f} "
            f"{row.get('accuracy', float('nan')):>8.4f} "
            f"{row.get('f1', float('nan')):>8.4f}"
        )
    print("=" * 90)


def t_critical_95(n: int) -> float:
    """Return t critical value for 95 % two-tailed CI."""
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, df=n - 1))
    except ImportError:
        table = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
        }
        df = n - 1
        if df in table:
            return table[df]
        return 1.960 if df > 30 else 2.0


def compute_ci(values: list[float]) -> dict:
    arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
    n = len(arr)
    if n == 0:
        nan = float("nan")
        return dict(n=0, mean=nan, std=nan, sem=nan, ci95_half=nan, ci95_low=nan, ci95_high=nan)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 else float("nan")
    t_val = t_critical_95(n)
    half = t_val * sem if n > 1 else float("nan")
    return dict(
        n=n,
        mean=mean,
        std=std,
        sem=sem,
        ci95_half=half,
        ci95_low=mean - half if n > 1 else float("nan"),
        ci95_high=mean + half if n > 1 else float("nan"),
    )
