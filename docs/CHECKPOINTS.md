# Pre-trained Checkpoints

This repository does not include model checkpoints due to their size. Download them separately and place them in your local environment.

## Download Links

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

**Status:** 🔒 **Restricted access during peer review**  
Models will be publicly released after paper acceptance.

**For reviewers:** Access available upon request. Contact: [your.email@domain.com]

### SSL Pre-trained Models

**Zenodo Archive:** https://doi.org/10.5281/zenodo.XXXXXXX (Update after upload)  
**Backup (Google Drive):** [Link for reviewer access - OPTIONAL]

| Model | Epochs | Size | Description |
|-------|--------|------|-------------|
| `mae_vit_small_399ep.pth` | 399 | ~83 MB | MAE pre-trained ViT-Small on 4-channel multispectral |
| `mae_vit_base_399ep.pth` | 399 | ~330 MB | MAE pre-trained ViT-Base on 4-channel multispectral |
| `moco_swin_tiny_79ep.ckpt` | 79 | ~765 MB | MoCo v3 pre-trained Swin-Tiny on multispectral |
| `moco_swin_tiny_199ep.ckpt` | 199 | ~765 MB | MoCo v3 pre-trained Swin-Tiny (longer training) |
| `fastsiam_swin.ckpt` | - | ~259 MB | FastSiam pre-trained Swin-Tiny |

### Downstream Trained Models (Optional)

Best checkpoints from downstream segmentation experiments.

**Location:** [Google Drive Link - TO BE ADDED]

Structure:
```
downstream_sequoia/checkpoints/{split}/{weight}_{frozen|unfrozen}/best_model.ckpt
downstream_rededge/checkpoints/{split}/{weight}_{frozen|unfrozen}/best_model.ckpt
```

## Installation Instructions

### Option 1: Manual Download (Google Drive)

1. Download checkpoints from the Google Drive link above
2. Place them in your local data directory:

```bash
# SSL checkpoints
mkdir -p data/ssl_pretrain/checkpoints
# Place downloaded .pth and .ckpt files here

# Or set environment variables
export SSL_MAE_VIT_CHECKPOINT=/path/to/mae_vit_small_399ep.pth
export SSL_MAE_VIT_BASE_CHECKPOINT=/path/to/mae_vit_base_399ep.pth
export SSL_MOCO_SWIN_CHECKPOINT=/path/to/moco_swin_tiny_79ep.ckpt
export SSL_FASTSIAM_CHECKPOINT=/path/to/fastsiam_swin.ckpt
```

### Option 2: Using gdown (Python package)

```bash
pip install gdown

# Download from Google Drive (replace FILE_ID with actual ID)
gdown https://drive.google.com/uc?id=FILE_ID -O mae_vit_small_399ep.pth
```

### Option 3: Using Zenodo (if uploaded there)

```bash
# Download using wget or curl
wget https://zenodo.org/record/XXXXX/files/mae_vit_small_399ep.pth
```

### Option 4: Using HuggingFace Hub (if uploaded there)

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="your-username/multispectral-ssl",
    filename="mae_vit_small_399ep.pth"
)
```

## Where to Place Checkpoints Locally

After downloading, place checkpoints according to your setup:

### Using Repository Structure
```
IEEE_acess_code/
├── data/
│   └── ssl_pretrain/
│       └── checkpoints/
│           ├── mae_vit_small_399ep.pth
│           ├── mae_vit_base_399ep.pth
│           ├── moco_swin_tiny_79ep.ckpt
│           └── fastsiam_swin.ckpt
```

### Using Environment Variables
Set these in your shell or `configs/paths.local.yaml`:

```bash
export SSL_MAE_VIT_CHECKPOINT=/path/to/mae_vit_small_399ep.pth
export SSL_MAE_VIT_BASE_CHECKPOINT=/path/to/mae_vit_base_399ep.pth
export SSL_MOCO_SWIN_CHECKPOINT=/path/to/moco_swin_tiny_79ep.ckpt
export SSL_FASTSIAM_CHECKPOINT=/path/to/fastsiam_swin.ckpt
```

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
