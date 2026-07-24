"""
Convert MAE pretrained checkpoint to timm ViT checkpoint.

This script:
1. Loads a MAE pretrained checkpoint (trained with 4 channels)
2. Extracts encoder weights and maps them to timm ViT structure
3. Optionally reduces channels (e.g. 4 -> 3)
4. Saves as timm-compatible checkpoint
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
import timm


def convert_mae_to_timm_vit(
    mae_checkpoint_path,
    output_path,
    timm_model_name="vit_base_patch16_224.augreg_in21k",
    model_name="vit_base_patch16_224",
    in_chans_mae=4,
    in_chans_timm=4,
    img_size=224,
    patch_size=16,
):
    """
    Convert MAE checkpoint to timm ViT checkpoint.

    Args:
        mae_checkpoint_path: Path to MAE checkpoint (.pth file)
        output_path: Path to save timm-compatible checkpoint
        timm_model_name: timm model id used as weight template
        model_name: Short name stored in output metadata
        in_chans_mae: Number of input channels used in MAE training
        in_chans_timm: Number of input channels for timm model
        img_size: Image size
        patch_size: Patch size
    """
    print(f"Loading MAE checkpoint from: {mae_checkpoint_path}")

    checkpoint = torch.load(mae_checkpoint_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        mae_state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        mae_state_dict = checkpoint["state_dict"]
    else:
        mae_state_dict = checkpoint

    print(f"MAE checkpoint keys (first 10): {list(mae_state_dict.keys())[:10]}")
    if "epoch" in checkpoint:
        print(f"MAE checkpoint epoch: {checkpoint['epoch']}")

    print(f"\nCreating timm ViT model: {timm_model_name}")
    timm_model = timm.create_model(
        timm_model_name,
        pretrained=False,
        num_classes=0,
        img_size=img_size,
        in_chans=in_chans_timm,
    )

    timm_state_dict = timm_model.state_dict()
    print(f"Timm model keys (first 10): {list(timm_state_dict.keys())[:10]}")

    converted_state_dict = OrderedDict()

    print("\nConverting weights...")

    mae_patch_key = "patch_embed.proj.weight"
    if mae_patch_key in mae_state_dict:
        mae_patch_weight = mae_state_dict[mae_patch_key]
        timm_patch_weight = timm_state_dict["patch_embed.proj.weight"]

        if in_chans_mae == in_chans_timm:
            if mae_patch_weight.shape == timm_patch_weight.shape:
                converted_state_dict["patch_embed.proj.weight"] = mae_patch_weight
                print(f"  [OK] Copied patch_embed.proj.weight: {mae_patch_weight.shape}")
            else:
                raise ValueError(
                    f"Shape mismatch: MAE {mae_patch_weight.shape} vs timm {timm_patch_weight.shape}"
                )
        elif in_chans_mae > in_chans_timm:
            converted_state_dict["patch_embed.proj.weight"] = mae_patch_weight[
                :, :in_chans_timm, :, :
            ]
            print(
                f"  [OK] Reduced patch_embed.proj.weight from {in_chans_mae} to {in_chans_timm} "
                f"channels: {mae_patch_weight.shape} -> "
                f"{converted_state_dict['patch_embed.proj.weight'].shape}"
            )
        else:
            raise ValueError(
                f"Cannot convert from {in_chans_mae} to {in_chans_timm} channels "
                "(MAE has fewer channels)"
            )

        mae_patch_bias_key = "patch_embed.proj.bias"
        if mae_patch_bias_key in mae_state_dict:
            converted_state_dict["patch_embed.proj.bias"] = mae_state_dict[mae_patch_bias_key]
            print("  [OK] Copied patch_embed.proj.bias")

    if "cls_token" in mae_state_dict:
        mae_cls_token = mae_state_dict["cls_token"]
        if mae_cls_token.shape == timm_state_dict["cls_token"].shape:
            converted_state_dict["cls_token"] = mae_cls_token
            print(f"  [OK] Copied cls_token: {mae_cls_token.shape}")
        else:
            print(
                f"  [WARN] Warning: cls_token shape mismatch: "
                f"{mae_cls_token.shape} vs {timm_state_dict['cls_token'].shape}"
            )

    if "pos_embed" in mae_state_dict:
        mae_pos_embed = mae_state_dict["pos_embed"]
        timm_pos_embed = timm_state_dict["pos_embed"]
        if mae_pos_embed.shape == timm_pos_embed.shape:
            converted_state_dict["pos_embed"] = mae_pos_embed
            print(f"  [OK] Copied pos_embed: {mae_pos_embed.shape}")
        else:
            print(
                f"  [WARN] Warning: pos_embed shape mismatch: "
                f"{mae_pos_embed.shape} vs {timm_pos_embed.shape}"
            )
            print("    Using timm's default pos_embed (will be randomly initialized)")

    block_indices = sorted(
        {
            int(k.split(".")[1])
            for k in timm_state_dict
            if k.startswith("blocks.") and k.split(".")[1].isdigit()
        }
    )
    print(f"\n  Converting {len(block_indices)} transformer blocks...")

    for i in block_indices:
        block_prefix_mae = f"blocks.{i}."
        block_keys = [k for k in mae_state_dict if k.startswith(block_prefix_mae)]
        for mae_key in block_keys:
            timm_key = mae_key
            if timm_key not in timm_state_dict:
                print(f"  [WARN] Warning: Key {timm_key} not found in timm model")
                continue
            if mae_state_dict[mae_key].shape == timm_state_dict[timm_key].shape:
                converted_state_dict[timm_key] = mae_state_dict[mae_key]
            else:
                print(
                    f"  [WARN] Warning: Shape mismatch for {timm_key}: "
                    f"{mae_state_dict[mae_key].shape} vs {timm_state_dict[timm_key].shape}"
                )
        if block_keys:
            copied = len([k for k in block_keys if k in converted_state_dict])
            print(f"    [OK] Block {i}: Copied {copied} parameters")

    if "norm.weight" in mae_state_dict:
        if mae_state_dict["norm.weight"].shape == timm_state_dict["norm.weight"].shape:
            converted_state_dict["norm.weight"] = mae_state_dict["norm.weight"]
            converted_state_dict["norm.bias"] = mae_state_dict["norm.bias"]
            print("  [OK] Copied norm layer")
        else:
            print("  [WARN] Warning: norm layer shape mismatch")

    missing_keys = set(timm_state_dict.keys()) - set(converted_state_dict.keys())
    if missing_keys:
        print(f"\n  [WARN] Warning: {len(missing_keys)} keys not converted (will use random init):")
        for key in list(missing_keys)[:5]:
            print(f"    - {key}")
        if len(missing_keys) > 5:
            print(f"    ... and {len(missing_keys) - 5} more")

    print("\nLoading converted weights into timm model...")
    missing, unexpected = timm_model.load_state_dict(converted_state_dict, strict=False)

    if missing:
        print(f"  [WARN] Missing keys (using random init): {len(missing)}")
    if unexpected:
        print(f"  [WARN] Unexpected keys (ignored): {len(unexpected)}")

    print(f"\nSaving converted checkpoint to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": timm_model.state_dict(),
            "model_name": model_name,
            "img_size": img_size,
            "in_chans": in_chans_timm,
            "patch_size": patch_size,
            "source": "mae_pretrained",
            "mae_checkpoint": str(mae_checkpoint_path),
            "mae_epoch": checkpoint.get("epoch"),
        },
        output_path,
    )

    print("[OK] Successfully converted and saved checkpoint!")
    print("\nSummary:")
    print(f"  - Converted {len(converted_state_dict)} weight tensors")
    print(f"  - Model: {model_name}")
    print(f"  - Channels: {in_chans_mae} -> {in_chans_timm}")
    print(f"  - Output saved to: {output_path}")

    return timm_model, converted_state_dict


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    default_in = repo_root / "vit_base_patch16_224_mae_399.pth"
    default_out = repo_root / "vit_base_patch16_224_mae_pretrained_imagenet_ini_399ep.pth"

    parser = argparse.ArgumentParser(description="Convert MAE checkpoint to timm ViT encoder.")
    parser.add_argument(
        "--mae_checkpoint",
        type=str,
        default=str(default_in),
        help="Path to full MAE .pth (encoder+decoder).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(default_out),
        help="Path for converted timm ViT encoder checkpoint.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "base"],
        default="base",
        help="ViT size to convert (default: base).",
    )
    parser.add_argument("--in_chans_mae", type=int, default=4)
    parser.add_argument("--in_chans_timm", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    args = parser.parse_args()

    if args.model_size == "base":
        timm_model_name = "vit_base_patch16_224.augreg_in21k"
        model_name = "vit_base_patch16_224"
    else:
        timm_model_name = "vit_small_patch16_224.augreg_in21k"
        model_name = "vit_small_patch16_224"

    convert_mae_to_timm_vit(
        mae_checkpoint_path=args.mae_checkpoint,
        output_path=args.output_path,
        timm_model_name=timm_model_name,
        model_name=model_name,
        in_chans_mae=args.in_chans_mae,
        in_chans_timm=args.in_chans_timm,
        img_size=args.img_size,
        patch_size=args.patch_size,
    )

    print("\n" + "=" * 60)
    print("Conversion complete. Use the output as SSL_MAE_VIT_BASE_CHECKPOINT.")
    print("=" * 60)
