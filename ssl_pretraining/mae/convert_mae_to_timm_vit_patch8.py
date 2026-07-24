"""
Convert MAE pretrained checkpoint to timm ViT checkpoint.

This script:
1. Loads a MAE pretrained checkpoint (trained with 4 channels)
2. Extracts encoder weights and maps them to timm ViT structure
3. Handles channel reduction from 4 to 3 channels
4. Saves as timm-compatible checkpoint
"""

import torch
import timm
from collections import OrderedDict
import sys
import os

# Add path to import MAE model
sys.path.append(os.path.dirname(__file__))
from models_mae_mod_imagenet_init import mae_vit_small_patch8_dec512d8b


def convert_mae_to_timm_vit(
    mae_checkpoint_path,
    output_path,
    in_chans_mae=4,
    in_chans_timm=4,
    img_size=224,
    patch_size=8
):
    """
    Convert MAE checkpoint to timm ViT checkpoint.
    
    Args:
        mae_checkpoint_path: Path to MAE checkpoint (.pth file)
        output_path: Path to save timm-compatible checkpoint
        in_chans_mae: Number of input channels used in MAE training (default: 4)
        in_chans_timm: Number of input channels for timm model (default: 4, matches MAE)
        img_size: Image size (default: 224)
        patch_size: Patch size (default: 16)
    """
    print(f"Loading MAE checkpoint from: {mae_checkpoint_path}")
    
    # Load MAE checkpoint
    # Note: weights_only=False needed for PyTorch 2.6+ when checkpoint contains non-weight objects
    checkpoint = torch.load(mae_checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        mae_state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        mae_state_dict = checkpoint['state_dict']
    else:
        mae_state_dict = checkpoint
    
    print(f"MAE checkpoint keys (first 10): {list(mae_state_dict.keys())[:10]}")
    
    # Create timm  model to get target structure
    print(f"\nCreating timm ViT model: vit_small_patch8_224")
    timm_model = timm.create_model(
        'vit_small_patch8_224',
        pretrained=False,
        num_classes=0,  # Remove classifier
        img_size=img_size,
        in_chans=in_chans_timm
    )
    
    timm_state_dict = timm_model.state_dict()
    print(f"Timm model keys (first 10): {list(timm_state_dict.keys())[:10]}")
    
    # Create new state dict for timm model
    converted_state_dict = OrderedDict()
    
    # Mapping from MAE keys to timm keys
    key_mappings = {
        'patch_embed.proj.weight': 'patch_embed.proj.weight',
        'patch_embed.proj.bias': 'patch_embed.proj.bias',
        'cls_token': 'cls_token',
        'pos_embed': 'pos_embed',
        'norm.weight': 'norm.weight',
        'norm.bias': 'norm.bias',
    }
    
    # Copy and convert weights
    print("\nConverting weights...")
    
    # 1. Patch embedding - copy directly (no channel reduction)
    mae_patch_key = 'patch_embed.proj.weight'
    if mae_patch_key in mae_state_dict:
        mae_patch_weight = mae_state_dict[mae_patch_key]  # Shape: (embed_dim, in_chans_mae, patch_size, patch_size)
        timm_patch_weight = timm_state_dict['patch_embed.proj.weight']  # Shape: (embed_dim, in_chans_timm, patch_size, patch_size)
        
        if in_chans_mae == in_chans_timm:
            # Direct copy if channels match
            if mae_patch_weight.shape == timm_patch_weight.shape:
                converted_state_dict['patch_embed.proj.weight'] = mae_patch_weight
                print(f"  ✓ Copied patch_embed.proj.weight: {mae_patch_weight.shape}")
            else:
                raise ValueError(f"Shape mismatch: MAE {mae_patch_weight.shape} vs timm {timm_patch_weight.shape}")
        elif in_chans_mae > in_chans_timm:
            # Take first N channels if MAE has more
            converted_state_dict['patch_embed.proj.weight'] = mae_patch_weight[:, :in_chans_timm, :, :]
            print(f"  ✓ Reduced patch_embed.proj.weight from {in_chans_mae} to {in_chans_timm} channels: {mae_patch_weight.shape} -> {converted_state_dict['patch_embed.proj.weight'].shape}")
        else:
            # MAE has fewer channels - pad or repeat (unusual case)
            
            raise ValueError(f"Cannot convert from {in_chans_mae} to {in_chans_timm} channels (MAE has fewer channels)")
        
        # Copy bias if exists
        mae_patch_bias_key = 'patch_embed.proj.bias'
        if mae_patch_bias_key in mae_state_dict:
            converted_state_dict['patch_embed.proj.bias'] = mae_state_dict[mae_patch_bias_key]
            print(f"  ✓ Copied patch_embed.proj.bias")
    
    # 2. CLS token
    if 'cls_token' in mae_state_dict:
        mae_cls_token = mae_state_dict['cls_token']
        if mae_cls_token.shape == timm_state_dict['cls_token'].shape:
            converted_state_dict['cls_token'] = mae_cls_token
            print(f"  ✓ Copied cls_token: {mae_cls_token.shape}")
        else:
            print(f"  ⚠ Warning: cls_token shape mismatch: {mae_cls_token.shape} vs {timm_state_dict['cls_token'].shape}")
    
    # 3. Positional embedding
    if 'pos_embed' in mae_state_dict:
        mae_pos_embed = mae_state_dict['pos_embed']
        timm_pos_embed = timm_state_dict['pos_embed']
        
        if mae_pos_embed.shape == timm_pos_embed.shape:
            converted_state_dict['pos_embed'] = mae_pos_embed
            print(f"  ✓ Copied pos_embed: {mae_pos_embed.shape}")
        else:
            # Interpolate if needed
            print(f"  ⚠ Warning: pos_embed shape mismatch: {mae_pos_embed.shape} vs {timm_pos_embed.shape}")
            print(f"    Using timm's default pos_embed (will be randomly initialized)")
    
    # 4. Transformer blocks
    num_blocks = len([k for k in timm_state_dict.keys() if k.startswith('blocks.') and '.norm' in k])
    print(f"\n  Converting {num_blocks} transformer blocks...")
    
    for i in range(num_blocks):
        block_prefix_mae = f'blocks.{i}.'
        block_prefix_timm = f'blocks.{i}.'
        
        # Copy all block parameters
        block_keys = [k for k in mae_state_dict.keys() if k.startswith(block_prefix_mae)]
        
        for mae_key in block_keys:
            timm_key = mae_key  # Keys should match
            if timm_key in timm_state_dict:
                if mae_state_dict[mae_key].shape == timm_state_dict[timm_key].shape:
                    converted_state_dict[timm_key] = mae_state_dict[mae_key]
                else:
                    print(f"  ⚠ Warning: Shape mismatch for {timm_key}: {mae_state_dict[mae_key].shape} vs {timm_state_dict[timm_key].shape}")
            else:
                print(f"  ⚠ Warning: Key {timm_key} not found in timm model")
        
        if block_keys:
            print(f"    ✓ Block {i}: Copied {len([k for k in block_keys if k in converted_state_dict])} parameters")
    
    # 5. Norm layer
    if 'norm.weight' in mae_state_dict:
        if mae_state_dict['norm.weight'].shape == timm_state_dict['norm.weight'].shape:
            converted_state_dict['norm.weight'] = mae_state_dict['norm.weight']
            converted_state_dict['norm.bias'] = mae_state_dict['norm.bias']
            print(f"  ✓ Copied norm layer")
        else:
            print(f"  ⚠ Warning: norm layer shape mismatch")
    
    # Fill in missing keys with timm defaults (or random init)
    missing_keys = set(timm_state_dict.keys()) - set(converted_state_dict.keys())
    if missing_keys:
        print(f"\n  ⚠ Warning: {len(missing_keys)} keys not converted (will use random init):")
        for key in list(missing_keys)[:5]:
            print(f"    - {key}")
        if len(missing_keys) > 5:
            print(f"    ... and {len(missing_keys) - 5} more")
    
    # Load converted weights into timm model
    print(f"\nLoading converted weights into timm model...")
    missing, unexpected = timm_model.load_state_dict(converted_state_dict, strict=False)
    
    if missing:
        print(f"  ⚠ Missing keys (using random init): {len(missing)}")
    if unexpected:
        print(f"  ⚠ Unexpected keys (ignored): {len(unexpected)}")
    
    # Save checkpoint
    print(f"\nSaving converted checkpoint to: {output_path}")
    torch.save({
        'model': timm_model.state_dict(),
        'model_name': 'vit_small_patch8_224',
        'img_size': img_size,
        'in_chans': in_chans_timm,
        'patch_size': patch_size,
        'source': 'mae_pretrained',
        'mae_checkpoint': mae_checkpoint_path
    }, output_path)
    
    print(f"✓ Successfully converted and saved checkpoint!")
    print(f"\nSummary:")
    print(f"  - Converted {len(converted_state_dict)} weight tensors")
    print(f"  - Channels: {in_chans_mae} -> {in_chans_timm} (no reduction)")
    print(f"  - Output saved to: {output_path}")
    
    return timm_model, converted_state_dict


if __name__ == '__main__':
    # Configuration
    mae_checkpoint_path = r"/home/thomaslf/icarus_data_zenodo/MAE/checkpoints_mae_vit_small_patch8_ratio75_no_scale02_1_from512x512_ratio1_1_dino/checkpoint-399.pth"
    output_path = r"//home/thomaslf/icarus_data_zenodo/MAE/checkpoints_mae_vit_small_patch8_ratio75_no_scale02_1_from512x512_ratio1_1_dino/vit_small_patch8_224_mae_pretrained_dino_ini_399ep.pth"
    
    # Convert
    model, state_dict = convert_mae_to_timm_vit(
        mae_checkpoint_path=mae_checkpoint_path,
        output_path=output_path,
        in_chans_mae=4,  # Your MAE was trained with 4 channels
        in_chans_timm=4,  # Keep 4 channels for timm ViT (no reduction)
        img_size=224,
        patch_size=8
    )
    
    print("\n" + "="*60)
    print("Conversion complete! You can now use the checkpoint with:")
    print("  1. Direct timm model loading")
    print("  2. SMP UNet with timm encoder")
    print("="*60)

