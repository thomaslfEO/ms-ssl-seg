# Pre-trained Checkpoints

This repository does not include model checkpoints due to their size. Download them separately and place them in your local environment.

## Download Links

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21532723.svg)](https://doi.org/10.5281/zenodo.21532723)

**Zenodo Archive:** [https://doi.org/10.5281/zenodo.21532723](https://doi.org/10.5281/zenodo.21532723)

**Status:** In revision.

### SSL Pre-trained Models

| Checkpoint File | SSL Method | Architecture | Epochs | Size | Description |
|----------------|-----------|--------------|--------|------|-------------|
| `mae_vit_small_399ep.pth` | MAE | ViT-Small (patch16) | 399 | ~83 MB | Masked Autoencoder pre-trained Vision Transformer (Small) encoder on MSUAV500K+N. Use with DPT decoder for segmentation. |
| `mae_vit_base_399ep.pth` | MAE | ViT-Base (patch16) | 399 | ~330 MB | Masked Autoencoder pre-trained Vision Transformer (Base) encoder on MSUAV500K+N. Use with DPT decoder for segmentation. |
| `moco_v3_swin_tiny_79ep.ckpt` | MoCo v3 | Swin-S3-Tiny-224 | 79 | ~765 MB | Momentum Contrast v3 pre-trained Swin Transformer (Tiny) encoder on MSUAV500K+N. Use with U-Net/DeepLabV3+ for segmentation. |
| `moco_v3_vit_small_199ep.ckpt` | MoCo v3 | ViT-Small (patch16) | 199 | ~765 MB | Momentum Contrast v3 pre-trained Vision Transformer (Small) encoder on MSUAV500K+N. Use with DPT decoder for segmentation. |

**File Naming Convention:**
- `{method}_{architecture}_{variant}_{epochs}ep.{ext}`
- Example: `moco_v3_swin_tiny_79ep.ckpt`
  - `moco_v3`: Self-supervised learning method (MoCo version 3)
  - `swin`: Architecture family (Swin Transformer)
  - `tiny`: Model size variant
  - `79ep`: Trained for 79 epochs
  - `.ckpt`: PyTorch Lightning checkpoint format

**Training Data:**
All models trained on **MSUAV500K+N** - MSUAV500K extended with self-collected Finnish agricultural field data (4 spectral bands: Green, Red, Red-Edge, NIR).

**Model Types:**
- **MAE (Masked Autoencoder)**: Encoder-only weights, optimized for reconstruction tasks, work well for dense prediction
- **MoCo v3 (Momentum Contrast)**: Full model checkpoint (includes projection heads), encoder extracted during loading

### Downstream Trained Models (Optional)

Best checkpoints from downstream segmentation experiments.

**Location:** [Google Drive Link - TO BE ADDED]

Structure:
```
downstream_sequoia/checkpoints/{split}/{weight}_{frozen|unfrozen}/best_model.ckpt
downstream_rededge/checkpoints/{split}/{weight}_{frozen|unfrozen}/best_model.ckpt
```

## Installation Instructions

### Download from Zenodo

Download checkpoints directly from Zenodo: [https://doi.org/10.5281/zenodo.21532723](https://doi.org/10.5281/zenodo.21532723)

**Using wget/curl:**

```bash
# Download using wget
wget https://zenodo.org/records/21532723/files/mae_vit_small_399ep.pth
wget https://zenodo.org/records/21532723/files/mae_vit_base_399ep.pth
wget https://zenodo.org/records/21532723/files/moco_v3_swin_tiny_79ep.ckpt
wget https://zenodo.org/records/21532723/files/moco_v3_vit_small_199ep.ckpt

# Or using curl
curl -O https://zenodo.org/records/21532723/files/mae_vit_small_399ep.pth
```

### Set Environment Variables

After downloading, configure checkpoint locations:

```bash
export SSL_MAE_VIT_CHECKPOINT=/path/to/mae_vit_small_399ep.pth
export SSL_MAE_VIT_BASE_CHECKPOINT=/path/to/mae_vit_base_399ep.pth
export SSL_MOCO_SWIN_CHECKPOINT=/path/to/moco_v3_swin_tiny_79ep.ckpt
export SSL_MOCO_VIT_CHECKPOINT=/path/to/moco_v3_vit_small_199ep.ckpt
```

Or edit `configs/paths.local.yaml` with your checkpoint paths.

## Checkpoint Details

### MAE ViT Checkpoints

**Training hyperparameters:**
- Input size: 224×224
- Patch size: 16×16
- Masking ratio: 0.75
- Optimizer: AdamW (lr=1.5e-4, weight decay=0.05)
- Batch size: 256
- Epochs: 400
- Warmup: 40 epochs

### MoCo v3 Swin Checkpoints

**Training hyperparameters:**
- Architecture: Swin-S3-Tiny-224
- Temperature: 0.07
- Momentum: 0.99
- Optimizer: AdamW (lr=1e-3, weight decay=0.1)
- Batch size: 256
- Epochs: 200
- Warmup: 10 epochs

## Reproducing Checkpoints

To train your own checkpoints from scratch:

```bash
# MAE ViT-Small
cd ssl_pretraining/mae
python main_pretrain_imagenet_init.py \
  --model vit_small_patch16 \
  --epochs 400 \
  --batch_size 256

# MoCo v3 Swin
cd ssl_pretraining/moco
python train_moco_swin_tiny.py \
  --epochs 200 \
  --batch_size 256 \
  --temperature 0.07
```

See [docs/ssl_pretraining.md](ssl_pretraining.md) for full training instructions.

## Citation

If you use these pre-trained models, please cite our work:

```bibtex
@article{your_paper_2026,
  title={Your Paper Title},
  author={Your Name},
  journal={IEEE Access},
  year={2026}
}
```

## License

The model checkpoints follow the same license as the code repository. See [LICENSE](../LICENSE) for details.
