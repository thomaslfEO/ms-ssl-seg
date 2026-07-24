#!/usr/bin/env python3
"""Training loop for DPT + ViT (small or base) over data splits and weight configs."""

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
    from core_deep_learning_trainer_mica_vit import dl_train
    from simple_feedback import SimpleFeedback
    from train_loop_common import run_training_grid
except ImportError as exc:
    print(f"Error importing required modules: {exc}")
    sys.exit(1)


VIT_PRESETS = {
    "small": {
        "backbone": "tu-vit_small_patch16_224.augreg_in21k",
        "checkpoints_subdir": "checkpoints_dpt_vit_small",
        "ssl_weight_name": "mae_vit_small",
        "label": "DPT + ViT-Small",
    },
    "base": {
        "backbone": "tu-vit_base_patch16_224.augreg_in21k",
        "checkpoints_subdir": "checkpoints_dpt_vit_base",
        "ssl_weight_name": "mae_vit_base",
        "label": "DPT + ViT-Base",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train DPT models with ViT encoders in a loop over data splits, "
            "pretrained weights (ImageNet / none / MAE SSL), and freeze options."
        )
    )
    parser.add_argument(
        "--vit_size",
        type=str,
        choices=sorted(VIT_PRESETS.keys()),
        default="small",
        help="ViT encoder size: small (patch16) or base (patch16).",
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
        help="Root folder containing split subfolders (e.g., 5p, 10p, 100p). "
             "If not provided, uses dataset-specific default.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Override timm backbone (defaults depend on --vit_size).",
    )
    parser.add_argument("--data_aug", action="store_true", default=True)
    parser.add_argument("--no_data_aug", dest="data_aug", action="store_false")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--lr_finder", action="store_true")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--class_weights_balanced", action="store_true", default=True)
    parser.add_argument("--no_class_weights", dest="class_weights_balanced", action="store_false")
    parser.add_argument("--normalization", action="store_true", default=True)
    parser.add_argument("--no_normalization", dest="normalization", action="store_false")
    parser.add_argument("--device", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device_numbers", type=int, default=1)
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    # Resolve splits root based on dataset
    if args.splits_root is None:
        if args.dataset == "sequoia":
            args.splits_root = str(sequoia_splits_root())
        else:
            args.splits_root = str(rededge_splits_root())

    preset = VIT_PRESETS[args.vit_size]
    backbone = args.backbone or preset["backbone"]

    weight_name_map = {
        0: "imagenet",
        1: "none",
        2: "sentinel2_toa_resnet18",
        3: "sentinel2_toa_resnet50",
        4: preset["ssl_weight_name"],
    }

    run_training_grid(
        dl_train,
        SimpleFeedback(),
        splits_root=args.splits_root,
        checkpoints_subdir=preset["checkpoints_subdir"],
        arch_index=5,
        backbone=backbone,
        pretrained_weights_list=[0, 1, 4],
        weight_name_map=weight_name_map,
        freeze_options=[True, False],
        checkpoint_path=args.checkpoint,
        data_aug=args.data_aug,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        lr=args.lr,
        tune=args.lr_finder,
        early_stop=args.early_stop,
        class_weights_balanced=args.class_weights_balanced,
        normalization=args.normalization,
        num_workers=args.num_workers,
        num_models=args.num_models,
        device=args.device,
        device_numbers=args.device_numbers,
        experiment_label=preset["label"],
    )


if __name__ == "__main__":
    main()
