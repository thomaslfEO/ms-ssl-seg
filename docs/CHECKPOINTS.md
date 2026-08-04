# Pre-trained Checkpoints

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21532723.svg)](https://doi.org/10.5281/zenodo.21532723)

**Download:** [https://doi.org/10.5281/zenodo.21532723](https://doi.org/10.5281/zenodo.21532723)

## Available Models

All models trained on MSUAV500K+N (4-channel multispectral UAV imagery).

| File | Method | Architecture | Epochs |
|------|--------|--------------|--------|
| `mae_vit_small_399ep.pth` | MAE | ViT-Small | 399 |
| `mae_vit_base_399ep.pth` | MAE | ViT-Base | 399 |
| `moco_v3_swin_s3.ckpt` | MoCo v3 | Swin-Tiny | 79 |
| `moco_v3_vit_small_199ep.ckpt` | MoCo v3 | ViT-Small | 199 |

## Setup

Download from Zenodo and configure paths:

**Environment variables:**
```bash
export SSL_MAE_VIT_CHECKPOINT=/path/to/mae_vit_small_399ep.pth
export SSL_MOCO_SWIN_CHECKPOINT=/path/to/moco_v3_swin_s3.ckpt
```

**Or** edit `configs/paths.local.yaml`.

See [ssl_pretraining.md](ssl_pretraining.md) for training details.
