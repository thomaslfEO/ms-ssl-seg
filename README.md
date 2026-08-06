# Self-supervised training for high-resolution close-range multispectral remote sensing imagery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.11366-b31b1b.svg)](https://arxiv.org/abs/2607.11366)
<!-- TODO: Add Zenodo badge when available -->

> Official implementation of "[Self-supervised training for high-resolution close-range multispectral remote sensing imagery](https://arxiv.org/abs/2607.11366)"  
> Under Review, 2026

---

## Installation

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate ms-ssl-seg

# Or pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## Pre-trained Models

<!-- TODO: Update Zenodo link when available -->
Download from Zenodo (available upon publication):

| File | Method | Architecture | Downstream Use |
|------|--------|--------------|----------------|
| `mae_vit_small_399ep.pth` | MAE | ViT-Small | DPT decoder |
| `mae_vit_base_399ep.pth` | MAE | ViT-Base | DPT decoder |
| `moco_v3_swin_s3.ckpt` | MoCo v3 | Swin-Tiny | U-Net |
| `moco_v3_vit_small_199ep.ckpt` | MoCo v3 | ViT-Small | DPT decoder |

Configure paths via environment variables or `configs/paths.local.yaml`:

```bash
export SSL_MAE_VIT_CHECKPOINT=/path/to/mae_vit_small_399ep.pth
export SSL_MOCO_SWIN_CHECKPOINT=/path/to/moco_v3_swin_s3.ckpt
```

## Data Setup

Expected structure for downstream experiments:

```
{Sequoia|Rededge}_train_loop/
├── 5p/    # 5% training data split
├── 10p/
├── 25p/
├── 50p/
├── 75p/
└── 100p/
    ├── train_files.csv
    ├── validation_files.csv
    ├── Normalize_Bands.csv
    ├── Summary_train_val.csv
    ├── images/
    └── labels/
```

Set paths:

```bash
export SEQUOIA_SPLITS_ROOT=/path/to/Sequoia_train_loop
export REDEDGE_SPLITS_ROOT=/path/to/Rededge_train_loop
```

## Usage

### 1. SSL Pre-training

**MAE (ViT):**

```bash
cd ssl_pretraining/mae
python mae_pretrain_vit.py \
  --data_path /path/to/ssl_chips \
  --model mae_vit_small_patch16 \
  --epochs 400
```

**MoCo v3 (Swin):**

```bash
cd ssl_pretraining/moco
python train_moco_swin_tiny.py \
  --train_data /path/to/ssl_chips/train \
  --val_data /path/to/ssl_chips/val \
  --max_epochs 200
```

**Convert MAE for downstream:**

```bash
cd ssl_pretraining/mae
python convert_mae_to_timm_vit.py \
  --mae_checkpoint checkpoint.pth \
  --output_path vit_small_encoder.pth
```

### 2. Downstream Training

**Single run:**

```bash
cd downstream/specdeepmap
python train.py --input_folder Sequoia_train_loop/100p
```

**Full experiment grid** (all splits × pretrained weights × frozen/unfrozen):

```bash
# U-Net + Swin
python train_loop.py --dataset sequoia

# DPT + ViT
python train_loop_dpt_vit.py --dataset sequoia --vit_size small

# Switch to RedEdge
python train_loop.py --dataset rededge
```

### 3. Evaluation

```bash
# Test all checkpoints from training loop
python testing_loop.py --preset swin --dataset sequoia

# Single checkpoint
python test.py \
  --test_data_csv validation_files.csv \
  --model_checkpoint model.ckpt
```

## Citation

```bibtex
@article{thomas2026selfsupervised,
  title={Self-supervised training for high-resolution close-range multispectral remote sensing imagery},
  author={Thomas, Leon-Friedrich and Änäkkälä, Mikko and Lajunen, Antti},
  journal={Under Review},
  year={2026},
  url={https://arxiv.org/abs/2607.11366}
}
```

**Dataset citation:**

```bibtex
@dataset{thomas2026uav,
  author={Thomas, Leon-Friedrich and Änäkkälä, Mikko and Lajunen, Antti},
  title={UAV Multispectral Imagery of Agricultural Fields in Finland},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.18233335}
}
```

## License

MIT License. SpecDeepMap components retain their original license (see `downstream/specdeepmap/LICENSE_specdeepmap.md`).
