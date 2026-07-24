#!/usr/bin/env python3
"""
Evaluate all checkpoints produced by train_loop.py / train_loop_dpt_vit.py.

Expected layout:
  <splits_root>/<checkpoints_subdir>/<split_name>/<weight>_{frozen|unfrozen}/*.ckpt

Writes a flat list of IoU metrics (one entry per split × weight × freeze) to:
  <splits_root>/test_results/<checkpoints_subdir>/iou_results.json
  <splits_root>/test_results/<checkpoints_subdir>/iou_results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

conda_env = os.environ.get("CONDA_PREFIX", "")
if conda_env and sys.platform == "win32":
    possible_paths = [
        os.path.join(conda_env, "Library", "bin"),
        os.path.join(conda_env, "bin"),
        os.path.join(conda_env, "Scripts"),
    ]
    current_path = os.environ.get("PATH", "")
    for gdal_path in possible_paths:
        if os.path.exists(gdal_path) and gdal_path not in current_path:
            current_path = gdal_path + os.pathsep + current_path
    os.environ["PATH"] = current_path

sys.path.insert(0, str(Path(__file__).parent))
import _repo_setup  # noqa: F401, E402
from repo_paths import sequoia_splits_root, rededge_splits_root  # noqa: E402

try:
    from osgeo import gdal  # noqa: F401
except ImportError as gdal_error:
    print(f"WARNING: GDAL import test failed: {gdal_error}")

try:
    from core_tester_focal import process_images_from_csv
    from simple_feedback import SimpleFeedback
    from testing_loop_utils import (
        WEIGHT_DISPLAY,
        discover_config_dirs,
        find_best_checkpoint,
        print_results_table,
        read_mean_metrics_from_csv,
        resolve_test_csv,
        run_test_on_checkpoint,
        write_results_csv,
        write_results_json,
    )
except ImportError as exc:
    print(f"Error importing required modules: {exc}")
    sys.exit(1)


CHECKPOINTS_PRESETS = {
    "swin": "checkpoints",
    "dpt_vit_small": "checkpoints_dpt_vit_small",
    "dpt_vit_base": "checkpoints_dpt_vit_base",
}


def evaluate_all_checkpoints(
    *,
    splits_root: str,
    checkpoints_subdir: str,
    test_data_csv: str | None,
    device: int,
    skip_existing: bool,
    feedback,
) -> list[dict]:
    checkpoints_root = os.path.join(splits_root, checkpoints_subdir)
    experiment_dirs = discover_config_dirs(checkpoints_root)

    if not experiment_dirs:
        raise FileNotFoundError(
            f"No experiment folders matching '<weight>_{{frozen|unfrozen}}' under {checkpoints_root}"
        )

    results: list[dict] = []

    print("=" * 60)
    print("Testing Loop — evaluate all checkpoints")
    print("=" * 60)
    print(f"Splits root:         {splits_root}")
    print(f"Checkpoints subdir:  {checkpoints_subdir}")
    print(f"Experiments found:   {len(experiment_dirs)}")
    print(f"Device:              {'GPU' if device == 1 else 'CPU'}")
    print("=" * 60)

    for idx, (split_name, weight_name, freeze_str, experiment_dir) in enumerate(
        experiment_dirs, 1
    ):
        print(
            f"\n[{idx}/{len(experiment_dirs)}] "
            f"Split: {split_name} | {weight_name} | {freeze_str}"
        )

        test_csv = resolve_test_csv(splits_root, split_name, test_data_csv)
        if test_csv is None:
            print(f"  SKIP — no test_files.csv or validation_files.csv for {split_name}")
            continue

        ckpt_path = find_best_checkpoint(experiment_dir)
        if ckpt_path is None:
            print(f"  SKIP — no .ckpt in {experiment_dir}")
            continue

        output_csv = os.path.join(experiment_dir, "test_results.csv")
        if skip_existing and os.path.isfile(output_csv):
            print(f"  Re-using {output_csv}")
            metrics = read_mean_metrics_from_csv(output_csv)
        else:
            print(f"  Checkpoint: {os.path.basename(ckpt_path)}")
            print(f"  Test CSV:   {test_csv}")
            metrics = run_test_on_checkpoint(
                process_images_from_csv,
                ckpt_path=ckpt_path,
                test_csv=test_csv,
                output_csv=output_csv,
                acc_device=device,
                feedback=feedback,
            )

        if metrics is None:
            print("  WARNING — no metrics returned, skipping.")
            continue

        results.append(
            {
                "split": split_name,
                "weight_name": weight_name,
                "weight_display": WEIGHT_DISPLAY.get(weight_name, weight_name),
                "freeze": freeze_str,
                "checkpoint": ckpt_path,
                "test_csv": test_csv,
                **metrics,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every checkpoint from train_loop.py and save IoU metrics "
            "per data split and weight configuration."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sequoia",
        choices=["sequoia", "rededge"],
        help="Dataset to use: sequoia or rededge (default: sequoia)",
    )
    parser.add_argument(
        "--splits_root",
        type=str,
        default=None,
        help="Root folder with split subfolders and checkpoint subdirectories. "
             "If not provided, uses dataset-specific default.",
    )
    parser.add_argument(
        "--checkpoints_subdir",
        type=str,
        default="checkpoints",
        help=(
            "Checkpoint folder name under splits_root "
            "(default: checkpoints; use checkpoints_dpt_vit_small/base for DPT)."
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(CHECKPOINTS_PRESETS.keys()),
        default=None,
        help="Shortcut for --checkpoints_subdir (swin | dpt_vit_small | dpt_vit_base).",
    )
    parser.add_argument(
        "--test_data_csv",
        type=str,
        default=None,
        help="Optional global test CSV used for every split.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=1,
        choices=[0, 1],
        help="Device: 0=cpu, 1=gpu (default: 1).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Reuse test_results.csv if already present in the experiment folder.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory for iou_results.json/csv "
            "(default: <splits_root>/test_results/<checkpoints_subdir>/)."
        ),
    )
    args = parser.parse_args()

    # Resolve splits root based on dataset
    if args.splits_root is None:
        if args.dataset == "sequoia":
            args.splits_root = str(sequoia_splits_root())
        else:
            args.splits_root = str(rededge_splits_root())

    if not os.path.isdir(args.splits_root):
        raise FileNotFoundError(f"Splits root not found: {args.splits_root}")

    checkpoints_subdir = (
        CHECKPOINTS_PRESETS[args.preset] if args.preset else args.checkpoints_subdir
    )
    output_dir = args.output_dir or os.path.join(
        args.splits_root, "test_results", checkpoints_subdir
    )
    os.makedirs(output_dir, exist_ok=True)

    feedback = SimpleFeedback()
    results = evaluate_all_checkpoints(
        splits_root=args.splits_root,
        checkpoints_subdir=checkpoints_subdir,
        test_data_csv=args.test_data_csv,
        device=args.device,
        skip_existing=args.skip_existing,
        feedback=feedback,
    )

    json_path = os.path.join(output_dir, "iou_results.json")
    csv_path = os.path.join(output_dir, "iou_results.csv")

    write_results_json(results, json_path)
    write_results_csv(results, csv_path)

    print(f"\nSaved {len(results)} result(s) to:")
    print(f"  {json_path}")
    print(f"  {csv_path}")

    print_results_table(results)
    print("\nDone.")


if __name__ == "__main__":
    main()
