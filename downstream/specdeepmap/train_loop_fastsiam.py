#!/usr/bin/env python3
"""
Standalone Deep Learning Trainer Script (FastSiam variant)
Same as train_loop, but only uses pretrained weights index 4.
"""

import argparse
import os
import sys
from pathlib import Path

# Fix OpenMP library conflict on Windows
# This is a common issue when multiple libraries use different OpenMP runtimes
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Disable SSL verification for HuggingFace downloads (corporate proxy / missing CA chain)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests as _requests
_orig_request = _requests.Session.request
def _no_ssl_request(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig_request(self, method, url, **kwargs)
_requests.Session.request = _no_ssl_request

# Fix GDAL DLL path on Windows - must be done BEFORE importing osgeo
# GDAL's C extension (_gdal.pyd) needs to find GDAL DLLs
conda_env = os.environ.get("CONDA_PREFIX", "")
if conda_env and sys.platform == "win32":
    # Add multiple possible GDAL DLL locations
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

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import _repo_setup  # noqa: F401, E402
from repo_paths import sequoia_splits_root, rededge_splits_root  # noqa: E402

# Test GDAL import before importing core modules
try:
    from osgeo import gdal
    # If successful, GDAL is available
except ImportError as gdal_error:
    print(f"WARNING: GDAL import test failed: {gdal_error}")
    print("This may cause issues. Continuing anyway...")

try:
    from core_deep_learning_trainer_mica_swin_fastsiam import dl_train
    from simple_feedback import SimpleFeedback
except ImportError as e:
    error_msg = str(e)
    print(f"Error importing required modules: {error_msg}")

    if "_gdal" in error_msg or "gdal" in error_msg.lower():
        print("\n" + "=" * 60)
        print("GDAL is not installed or not properly configured.")
        print("=" * 60)
        print("\nTroubleshooting steps:")
        print("1. Check if GDAL is installed:")
        print("   conda list gdal")
        print("\n2. Try reinstalling GDAL:")
        print("   conda remove gdal --force")
        print("   conda install -c conda-forge gdal python-gdal")
        print("\n3. Or try a specific version:")
        print("   conda install -c conda-forge gdal=3.7.*")
        print("\n4. Test GDAL import:")
        print('   python -c "from osgeo import gdal; print(gdal.__version__)"')
        print("\n5. If still failing, try:")
        print("   conda install -c conda-forge gdal libgdal")
        print("\nNote: On Windows, GDAL installation can be complex.")
        print("You may need to restart your terminal after installation.")
        print("=" * 60)
    else:
        print("Make sure you're running this from the SpecDeepMap directory")
        print("and all required dependencies are installed.")
    sys.exit(1)


def run_single_experiment(
    input_folder,
    arch_index,
    backbone,
    pretrained_weights_index,
    checkpoint_path,
    freeze_encoder,
    data_aug,
    batch_size,
    n_epochs,
    lr,
    tune,
    early_stop,
    class_weights_balanced,
    normalization_bool,
    num_workers,
    num_models,
    acc_type_index,
    acc_type_numbers,
    logdirpath,
    logdirpath_model,
    feedback,
):
    """Run a single dl_train experiment with proper cleanup."""
    print("\n" + "-" * 60)
    print(f"Input folder: {input_folder}")
    print(f"Architecture index: {arch_index}")
    print(f"Backbone: {backbone}")
    print(f"Pretrained weights index: {pretrained_weights_index}")
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Device: {'GPU' if acc_type_index == 1 else 'CPU'}")
    print(f"Logdir / checkpoints: {logdirpath_model}")
    print("-" * 60)

    model = None
    try:
        model = dl_train(
            input_folder=input_folder,
            arch_index=arch_index,
            backbone=backbone,
            pretrained_weights_index=pretrained_weights_index,
            checkpoint_path=checkpoint_path,
            freeze_encoder=freeze_encoder,
            data_aug=data_aug,
            batch_size=batch_size,
            n_epochs=n_epochs,
            lr=lr,
            tune=tune,
            early_stop=early_stop,
            class_weights_balanced=class_weights_balanced,
            normalization_bool=normalization_bool,
            num_workers=num_workers,
            num_models=num_models,
            acc_type_index=acc_type_index,
            acc_type_numbers=acc_type_numbers,
            logdirpath=logdirpath,
            logdirpath_model=logdirpath_model,
            feedback=feedback,
        )
        feedback.pushInfo("Training completed successfully!")
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
    except KeyboardInterrupt:
        feedback.pushInfo("Training interrupted by user")
        print("\nTraining interrupted by user")
        raise
    except Exception as e:
        feedback.pushError(f"Training failed: {str(e)}")
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Ensure model is deleted and CUDA cache is cleared so runs
        # never reuse any previous model accidentally.
        try:
            if model is not None:
                del model
        except NameError:
            pass

        try:
            import torch

            if acc_type_index == 1:
                torch.cuda.empty_cache()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(
        description=(
            "FastSiam training loop: runs over data splits, "
            "using only pretrained weights index 4 with frozen/unfrozen encoder."
        )
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="sequoia",
        choices=["sequoia", "rededge"],
        help="Dataset to use: sequoia or rededge (default: sequoia)",
    )

    # Root with all train_p* splits
    parser.add_argument(
        "--splits_root",
        type=str,
        default=None,
        help="Root folder containing split subfolders (e.g., 5p, 10p, 100p). "
             "If not provided, uses dataset-specific default.",
    )

    # Model architecture (defaults to U-Net + Swin S3 Tiny 224)
    parser.add_argument(
        "--arch",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help=(
            "Model architecture: 0=U-Net, 1=U-Net++, 2=DeepLabV3+, 3=SegFormer, "
            "4=2D-Justo-UNet-Simple, 5=DPT (default: 0)"
        ),
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="tu-swin_s3_tiny_224",
        help="Model backbone (default: tu-swin_s3_tiny_224)",
    )

    # Training parameters (shared across all runs)
    parser.add_argument(
        "--data_aug",
        action="store_true",
        default=True,
        help="Enable data augmentation (default: True)",
    )
    parser.add_argument(
        "--no_data_aug",
        dest="data_aug",
        action="store_false",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="Learning rate (default: 0.0003)",
    )
    parser.add_argument(
        "--lr_finder",
        action="store_true",
        help="Use automatic learning rate finder",
    )

    # Training options
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="Enable early stopping",
    )
    parser.add_argument(
        "--class_weights_balanced",
        action="store_true",
        default=True,
        help="Use balanced class weights (default: True)",
    )
    parser.add_argument(
        "--no_class_weights",
        dest="class_weights_balanced",
        action="store_false",
        help="Do not use balanced class weights",
    )
    parser.add_argument(
        "--normalization",
        action="store_true",
        default=True,
        help="Enable data normalization (default: True)",
    )
    parser.add_argument(
        "--no_normalization",
        dest="normalization",
        action="store_false",
        help="Disable data normalization",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=int,
        default=1,
        choices=[0, 1],
        help="Device: 0=cpu, 1=gpu (default: 1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading (default: 0)",
    )
    parser.add_argument(
        "--device_numbers",
        type=int,
        default=1,
        help="Number of devices (GPUs/CPUs) to use (default: 1)",
    )

    # Model saving
    parser.add_argument(
        "--num_models",
        type=int,
        default=1,
        help="Number of models to save (-1 means save each epoch, default: 1)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to continue training (optional)",
    )

    args = parser.parse_args()

    # Resolve splits root based on dataset
    if args.splits_root is None:
        if args.dataset == "sequoia":
            args.splits_root = str(sequoia_splits_root())
        else:
            args.splits_root = str(rededge_splits_root())

    # Only pretrained weights index 4, with frozen/unfrozen encoder
    pretrained_weights_list = [4]
    freeze_options = [True, False]

    weight_name_map = {
        0: "imagenet",
        1: "none",
        2: "sentinel2_toa_resnet18",
        3: "sentinel2_toa_resnet50",
        4: "micasense_sr_swin_s3_tiny",
    }

    splits_root = args.splits_root
    if not os.path.isdir(splits_root):
        raise FileNotFoundError(f"Splits root folder not found: {splits_root}")

    # Find all train_p* subfolders
    split_names = sorted(
        d
        for d in os.listdir(splits_root)
        if os.path.isdir(os.path.join(splits_root, d)) and d.startswith("train_p")
    )
    if not split_names:
        raise RuntimeError(f"No 'train_p*' folders found under {splits_root}")

    # Example target:
    # C:/Sequoia/label_train_val_loop/checkpoints/train_p5/imagenet_frozen
    checkpoints_root = os.path.join(splits_root, "checkpoints")

    # Shared feedback object across runs
    feedback = SimpleFeedback()

    print("=" * 60)
    print("Starting FastSiam Deep Learning Training Loop")
    print("=" * 60)
    print(f"Splits root: {splits_root}")
    print(f"Checkpoints root: {checkpoints_root}")
    print(f"Architecture index: {args.arch}")
    print(f"Backbone: {args.backbone}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {'GPU' if args.device == 1 else 'CPU'}")
    print(f"Pretrained weights indices: {pretrained_weights_list}")
    print(f"Freeze encoder options: {freeze_options}")
    print(f"Data splits: {', '.join(split_names)}")
    print("=" * 60)

    # Loop over splits, pretrained weights and freeze options
    for split_name in split_names:
        input_folder = os.path.join(splits_root, split_name)

        for pw_idx in pretrained_weights_list:
            weight_name = weight_name_map.get(pw_idx, f"weights_{pw_idx}")

            for freeze in freeze_options:
                freeze_str = "frozen" if freeze else "unfrozen"
                # Append 'fastsiam' so checkpoints don't clash with original index-4 saves
                run_subdir = f"{weight_name}_fastsiam_{freeze_str}"

                # Example:
                # C:/Sequoia/label_train_val_loop/checkpoints/train_p5/micasense_sr_swin_s3_tiny_fastsiam_frozen
                experiment_dir = os.path.join(
                    checkpoints_root,
                    split_name,
                    run_subdir,
                )
                os.makedirs(experiment_dir, exist_ok=True)

                print(
                    f"\n>>> Split: {split_name} | "
                    f"weights: {weight_name} ({pw_idx}) | "
                    f"freeze_encoder: {freeze_str}"
                )

                run_single_experiment(
                    input_folder=input_folder,
                    arch_index=args.arch,
                    backbone=args.backbone,
                    pretrained_weights_index=pw_idx,
                    checkpoint_path=args.checkpoint,
                    freeze_encoder=freeze,
                    data_aug=args.data_aug,
                    batch_size=args.batch_size,
                    n_epochs=args.n_epochs,
                    lr=args.lr,
                    tune=args.lr_finder,
                    early_stop=args.early_stop,
                    class_weights_balanced=args.class_weights_balanced,
                    normalization_bool=args.normalization,
                    num_workers=args.num_workers,
                    num_models=args.num_models,
                    acc_type_index=args.device,
                    acc_type_numbers=args.device_numbers,
                    logdirpath=experiment_dir,
                    logdirpath_model=experiment_dir,
                    feedback=feedback,
                )

    print("\n" + "=" * 60)
    print("All FastSiam looped trainings finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()

