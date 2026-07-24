#!/usr/bin/env python3
"""
Modify learning rate in checkpoint metadata.
This only changes the hyperparameters metadata, not the trained weights.
"""

import torch
import sys
import shutil
from pathlib import Path

def modify_checkpoint_lr(checkpoint_path, new_lr, backup=True):
    """
    Modify the learning rate in a checkpoint's hyperparameters.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        new_lr: New learning rate value (e.g., 0.0001)
        backup: Whether to create a backup of the original checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Create backup
    if backup:
        backup_path = checkpoint_path.with_suffix('.ckpt.backup')
        print(f"Creating backup: {backup_path}")
        shutil.copy2(checkpoint_path, backup_path)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if hyper_parameters exists
    if 'hyper_parameters' not in ckpt:
        print("Error: No 'hyper_parameters' key found in checkpoint")
        sys.exit(1)
    
    # Display current LR
    old_lr = ckpt['hyper_parameters'].get('lr', 'Not found')
    print(f"\nCurrent LR: {old_lr}")
    print(f"New LR:     {new_lr}")
    
    # Modify LR
    ckpt['hyper_parameters']['lr'] = new_lr
    
    # Save modified checkpoint
    print(f"\nSaving modified checkpoint...")
    torch.save(ckpt, checkpoint_path)
    
    # Verify
    print("\nVerifying modification...")
    ckpt_verify = torch.load(checkpoint_path, map_location='cpu')
    verified_lr = ckpt_verify['hyper_parameters']['lr']
    
    if verified_lr == new_lr:
        print(f"[OK] Success! LR changed from {old_lr} to {verified_lr}")
    else:
        print(f"[WARN] Warning: Verification failed. LR is {verified_lr}, expected {new_lr}")
    
    print(f"\nOriginal checkpoint backed up to: {backup_path}" if backup else "")
    print("Done!")


if __name__ == "__main__":
    checkpoint_path = r"C:\Users\leon-\Desktop\IEEE_acess_code\moco_v3_vit_small_epoch=199_train_train_loss_epoch=0.1233_val_val_loss_epoch=0.2468.ckpt"
    new_lr = 0.0001
    
    print("=" * 60)
    print("Checkpoint LR Modifier")
    print("=" * 60)
    
    modify_checkpoint_lr(checkpoint_path, new_lr, backup=True)
