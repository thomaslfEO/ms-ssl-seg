#!/usr/bin/env python3
"""
Standalone Deep Learning Trainer Script
This script allows training models without QGIS/enmapbox dependencies.
"""

import argparse
import os
import sys
from pathlib import Path

# Fix OpenMP library conflict on Windows
# This is a common issue when multiple libraries use different OpenMP runtimes
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Fix GDAL DLL path on Windows - must be done BEFORE importing osgeo
# GDAL's C extension (_gdal.pyd) needs to find GDAL DLLs
conda_env = os.environ.get('CONDA_PREFIX', '')
if conda_env and sys.platform == 'win32':
    # Add multiple possible GDAL DLL locations
    possible_paths = [
        os.path.join(conda_env, 'Library', 'bin'),
        os.path.join(conda_env, 'bin'),
        os.path.join(conda_env, 'Scripts'),
    ]
    current_path = os.environ.get('PATH', '')
    for gdal_path in possible_paths:
        if os.path.exists(gdal_path) and gdal_path not in current_path:
            current_path = gdal_path + os.pathsep + current_path
    os.environ['PATH'] = current_path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import _repo_setup  # noqa: F401, E402
from repo_paths import downstream_output_dir, sequoia_splits_root  # noqa: E402

# Test GDAL import before importing core modules
try:
    from osgeo import gdal
    # If successful, GDAL is available
except ImportError as gdal_error:
    print(f"WARNING: GDAL import test failed: {gdal_error}")
    print("This may cause issues. Continuing anyway...")

try:
    from core_deep_learning_trainer_mica_swin import dl_train
    from simple_feedback import SimpleFeedback
except ImportError as e:
    error_msg = str(e)
    print(f"Error importing required modules: {error_msg}")
    
    if '_gdal' in error_msg or 'gdal' in error_msg.lower():
        print("\n" + "="*60)
        print("GDAL is not installed or not properly configured.")
        print("="*60)
        print("\nTroubleshooting steps:")
        print("1. Check if GDAL is installed:")
        print("   conda list gdal")
        print("\n2. Try reinstalling GDAL:")
        print("   conda remove gdal --force")
        print("   conda install -c conda-forge gdal python-gdal")
        print("\n3. Or try a specific version:")
        print("   conda install -c conda-forge gdal=3.7.*")
        print("\n4. Test GDAL import:")
        print("   python -c \"from osgeo import gdal; print(gdal.__version__)\"")
        print("\n5. If still failing, try:")
        print("   conda install -c conda-forge gdal libgdal")
        print("\nNote: On Windows, GDAL installation can be complex.")
        print("You may need to restart your terminal after installation.")
        print("="*60)
    else:
        print("Make sure you're running this from the SpecDeepMap directory")
        print("and all required dependencies are installed.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train a deep learning model for semantic segmentation')
    
    # Required arguments (with defaults for convenience)
    parser.add_argument('--input_folder', type=str,
                        default=str(sequoia_splits_root() / "train_p100"),
                        help='Input folder containing train/validation datasets (must have train_files.csv, validation_files.csv, Summary_train_val.csv)')
    parser.add_argument('--logdirpath', type=str,
                        default=str(downstream_output_dir("sequoia") / "logs"),
                        help='Path for saving Tensorboard logger')
    parser.add_argument('--logdirpath_model', type=str,
                        default=str(downstream_output_dir("sequoia") / "checkpoints"),
                        help='Path for saving model checkpoints')
    
    # Model architecture
    parser.add_argument('--arch', type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5],
                        help='Model architecture: 0=U-Net, 1=U-Net++, 2=DeepLabV3+, 3=SegFormer, 4=2D-Justo-UNet-Simple, 5=DPT (default: 0)')
    parser.add_argument('--backbone', type=str, default='tu-swin_s3_tiny_224',
                        help='Model backbone (default: tu-vit_small_patch16_224.augreg_in21k)')
    
    # Pretrained weights
    parser.add_argument('--pretrained_weights', type=int, default=4,
                        choices=[0, 1, 2, 3, 4],
                        help='Pretrained weights: 0=imagenet, 1=None, 2=Sentinel_2_TOA_Resnet18, 3=Sentinel_2_TOA_Resnet50, 4=MicaSense_SR_Swin_s3_tiny (default: 4)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file to continue training (optional)')
    
    # Training parameters
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='Freeze backbone encoder (default: True)')
   
    parser.add_argument('--data_aug', action='store_true', default=True,
                        help='Enable data augmentation (default: True)')
    parser.add_argument('--no_data_aug', dest='data_aug', action='store_false',
                        help='Disable data augmentation')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate (default: 0.0003)')
    parser.add_argument('--lr_finder', action='store_true',
                        help='Use automatic learning rate finder')
    
    # Training options
    parser.add_argument('--early_stop', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--class_weights_balanced', action='store_true', default=True,
                        help='Use balanced class weights (default: True)')
    parser.add_argument('--no_class_weights', dest='class_weights_balanced', action='store_false',
                        help='Do not use balanced class weights')
    parser.add_argument('--normalization', action='store_true', default=True,
                        help='Enable data normalization (default: True)')
    parser.add_argument('--no_normalization', dest='normalization', action='store_false',
                        help='Disable data normalization')
    
    # Hardware
    parser.add_argument('--device', type=int, default=1,
                        choices=[0, 1],
                        help='Device: 0=cpu, 1=gpu (default: 1)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading (default: 0)')
    parser.add_argument('--device_numbers', type=int, default=1,
                        help='Number of devices (GPUs/CPUs) to use (default: 1)')
    
    # Model saving
    parser.add_argument('--num_models', type=int, default=1,
                        help='Number of models to save (-1 means save each epoch, default: -1)')
    
    args = parser.parse_args()
    
    # Create feedback object
    feedback = SimpleFeedback()
    
    print("=" * 60)
    print("Starting Deep Learning Training")
    print("=" * 60)
    print(f"Input folder: {args.input_folder}")
    print(f"Architecture index: {args.arch}")
    print(f"Backbone: {args.backbone}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {'GPU' if args.device == 1 else 'CPU'}")
    print("=" * 60)
    
    model = None
    try:
        model = dl_train(
            input_folder=args.input_folder,
            arch_index=args.arch,
            backbone=args.backbone,
            pretrained_weights_index=args.pretrained_weights,
            checkpoint_path=args.checkpoint,
            freeze_encoder=args.freeze_encoder,
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
            logdirpath=args.logdirpath,
            logdirpath_model=args.logdirpath_model,
            feedback=feedback
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
        # Ensure model is deleted and CUDA cache is cleared so a new run
        # never reuses any previous model accidentally.
        try:
            if model is not None:
                del model
        except NameError:
            pass

        try:
            import torch
            if args.device == 1:
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == '__main__':
    main()
