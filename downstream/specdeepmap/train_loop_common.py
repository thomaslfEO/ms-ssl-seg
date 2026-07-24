"""Shared grid-training utilities for downstream experiment loops."""

from __future__ import annotations

import os
from typing import Callable


def discover_split_names(splits_root: str) -> list[str]:
    """Discover split directories.
    
    Supports two naming conventions:
    1. Legacy: train_p5, train_p10, train_p25, etc.
    2. New: 5p, 10p, 25p, 50p, 75p, 100p
    """
    if not os.path.isdir(splits_root):
        raise FileNotFoundError(f"Splits root folder not found: {splits_root}")

    split_names = []
    for name in os.listdir(splits_root):
        path = os.path.join(splits_root, name)
        if not os.path.isdir(path):
            continue
        
        # Legacy format: train_p*
        if name.startswith("train_p"):
            split_names.append(name)
        # New format: 5p, 10p, 25p, etc.
        elif name.endswith("p") and name[:-1].isdigit():
            split_names.append(name)
    
    if not split_names:
        raise RuntimeError(
            f"No split folders found under {splits_root}\n"
            f"Expected 'train_p*' or '*p' format (e.g., train_p100 or 100p)"
        )
    
    # Sort numerically: extract percentage value
    def extract_percent(s: str) -> int:
        if s.startswith("train_p"):
            return int(s.replace("train_p", "").replace("p", ""))
        else:
            return int(s.replace("p", ""))
    
    return sorted(split_names, key=extract_percent)


def run_single_experiment(
    dl_train: Callable,
    feedback,
    *,
    input_folder: str,
    arch_index: int,
    backbone: str,
    pretrained_weights_index: int,
    checkpoint_path: str | None,
    freeze_encoder: bool,
    data_aug: bool,
    batch_size: int,
    n_epochs: int,
    lr: float,
    tune: bool,
    early_stop: bool,
    class_weights_balanced: bool,
    normalization_bool: bool,
    num_workers: int,
    num_models: int,
    acc_type_index: int,
    acc_type_numbers: int,
    logdirpath: str,
    logdirpath_model: str,
) -> None:
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
    except Exception as exc:
        feedback.pushError(f"Training failed: {str(exc)}")
        print(f"\nError during training: {str(exc)}")
        import traceback

        traceback.print_exc()
        raise
    finally:
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


def run_training_grid(
    dl_train: Callable,
    feedback,
    *,
    splits_root: str,
    checkpoints_subdir: str,
    arch_index: int,
    backbone: str,
    pretrained_weights_list: list[int],
    weight_name_map: dict[int, str],
    freeze_options: list[bool],
    checkpoint_path: str | None,
    data_aug: bool,
    batch_size: int,
    n_epochs: int,
    lr: float,
    tune: bool,
    early_stop: bool,
    class_weights_balanced: bool,
    normalization: bool,
    num_workers: int,
    num_models: int,
    device: int,
    device_numbers: int,
    experiment_label: str,
) -> None:
    split_names = discover_split_names(splits_root)
    checkpoints_root = os.path.join(splits_root, checkpoints_subdir)

    print("=" * 60)
    print(f"Starting Deep Learning Training Loop ({experiment_label})")
    print("=" * 60)
    print(f"Splits root: {splits_root}")
    print(f"Checkpoints root: {checkpoints_root}")
    print(f"Architecture index: {arch_index}")
    print(f"Backbone: {backbone}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Device: {'GPU' if device == 1 else 'CPU'}")
    print(f"Pretrained weights indices: {pretrained_weights_list}")
    print(f"Freeze encoder options: {freeze_options}")
    print(f"Data splits: {', '.join(split_names)}")
    print("=" * 60)

    for split_name in split_names:
        input_folder = os.path.join(splits_root, split_name)

        for pw_idx in pretrained_weights_list:
            weight_name = weight_name_map.get(pw_idx, f"weights_{pw_idx}")

            for freeze in freeze_options:
                freeze_str = "frozen" if freeze else "unfrozen"
                experiment_dir = os.path.join(checkpoints_root, split_name, f"{weight_name}_{freeze_str}")
                os.makedirs(experiment_dir, exist_ok=True)

                print(
                    f"\n>>> Split: {split_name} | "
                    f"weights: {weight_name} ({pw_idx}) | "
                    f"freeze_encoder: {freeze_str}"
                )

                run_single_experiment(
                    dl_train,
                    feedback,
                    input_folder=input_folder,
                    arch_index=arch_index,
                    backbone=backbone,
                    pretrained_weights_index=pw_idx,
                    checkpoint_path=checkpoint_path,
                    freeze_encoder=freeze,
                    data_aug=data_aug,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    lr=lr,
                    tune=tune,
                    early_stop=early_stop,
                    class_weights_balanced=class_weights_balanced,
                    normalization_bool=normalization,
                    num_workers=num_workers,
                    num_models=num_models,
                    acc_type_index=device,
                    acc_type_numbers=device_numbers,
                    logdirpath=experiment_dir,
                    logdirpath_model=experiment_dir,
                )

    print("\n" + "=" * 60)
    print("All looped trainings finished.")
    print("=" * 60)
