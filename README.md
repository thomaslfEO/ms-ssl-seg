# Self-supervised training for high-resolution close-range multispectral remote sensing imagery

> **Official implementation** (Under Review, 2026)

Self-supervised pretraining (**MAE**, **MoCo v3**) on 4-channel UAV-based multispectral chips from **MSUAV500K** and self-collected data from Finnish agricultural fields, with downstream semantic segmentation on the **WeedMap** dataset using **Sequoia** and **RedEdge** sensors.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.11366-b31b1b.svg)](https://arxiv.org/abs/2607.11366)

## 📄 Paper Information

**Status:** Under Review  
**Preprint:** [arXiv:2607.11366](https://arxiv.org/abs/2607.11366)

### Pre-trained Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21532723.svg)](https://doi.org/10.5281/zenodo.21532723)

All pre-trained SSL checkpoints are archived on **Zenodo**: [https://doi.org/10.5281/zenodo.21532723](https://doi.org/10.5281/zenodo.21532723)

**Status:** In revision.

See **[docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)** for details and download instructions.

### Dataset

A portion of the self-collected training data used for this work is publicly available:

**Thomas, L.-F., Änäkkälä, M. & Lajunen, A. (2026).** UAV Multispectral Imagery of Agricultural Fields in Finland [Dataset]. *Zenodo*. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18233335.svg)](https://doi.org/10.5281/zenodo.18233335)

This dataset contains 10-meter resolution UAV multispectral imagery from agricultural fields in Finland. It represents a portion of the self-collected data used for model training and extends the MSUAV500K dataset.

## Repository layout

```text
ssl_pretraining/     MAE and MoCo training scripts
preprocessing/       Dataset filtering, splits, summary CSVs
downstream/          SpecDeepMap segmentation trainer and experiment loops
splits/              Train/val/test CSV manifests (no rasters)
configs/             Example path configuration
environments/        Conda environment files
data/                Local data root (gitignored)
outputs/             Checkpoints and logs (gitignored)
```

## Setup

### 1. Environment Setup

**Option A: Conda (Recommended)**

```bash
conda env create -f environment.yml
conda activate ms-ssl-seg
```

**Option B: Pip with GPU (CUDA 12.4)**

```bash
conda create -n ms-ssl-seg python=3.11
conda activate ms-ssl-seg

# Install PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install GDAL via conda (recommended for Windows)
conda install -c conda-forge gdal rasterio libgdal

# Install remaining dependencies
pip install -r requirements.txt
```

For other CUDA versions, see: https://pytorch.org/get-started/locally/

**Option C: CPU-only**

Edit `environment.yml` and replace the torch pip lines with:
```
torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
Then run `conda env create -f environment.yml`

**Legacy environment:** `environments/specdeepmap_gpu_time_capsul.yml` (optional, pinned versions)

### 2. Configuration

Configure paths using **either** method:

**Option A:** Environment variables
```bash
export DATA_ROOT=/path/to/your/data
export OUTPUT_ROOT=/path/to/your/outputs
export SSL_MOCO_SWIN_CHECKPOINT=/path/to/moco_swin.ckpt
```

**Option B:** Config file
```bash
cp configs/paths.example.yaml configs/paths.local.yaml
# Edit paths.local.yaml with your paths
```

### 3. Pre-trained Checkpoints (Optional)

Download from Zenodo: **[docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)**

Required only for downstream experiments with SSL pretraining.

## Data Layout

**Note:** GeoTIFF datasets are not included in this repository (too large for GitHub). Only split manifests (CSV files) are provided in `splits/`.

**See [docs/DATA_STRUCTURE.md](docs/DATA_STRUCTURE.md) for detailed data organization guidelines.**

Place your datasets under `DATA_ROOT`:

```text
data/
├── ssl_pretrain/chips_512_4ch/
│   ├── train/all/              # unlabeled .tif chips for SSL
│   └── val/all/
├── Sequoia_train_loop/         # Downstream experiments
│   ├── 5p/
│   ├── 10p/
│   ├── 25p/
│   ├── 50p/
│   ├── 75p/
│   └── 100p/
└── Rededge_train_loop/         # Downstream experiments
    ├── 5p/
    ├── 10p/
    ├── 25p/
    ├── 50p/
    ├── 75p/
    └── 100p/
```

Each downstream split folder (5p, 10p, etc.) contains:
- `train_files.csv`, `validation_files.csv`: Image/mask pairs
- `Normalize_Bands.csv`: Per-split band normalization (mean, std)
- `Summary_train_val.csv`: Per-split class statistics and weights
- `images/`, `labels/`: Tile directories (or referenced in CSVs)

## Pipeline

### 1. Preprocessing (Sequoia)

```bash
python preprocessing/sequoia/filter_chips_by_label.py --image_dir ... --label_dir ... --delete
python preprocessing/sequoia/generate_accumulative_splits.py --base_dir $DATA_ROOT/sequoia/filtered
python preprocessing/sequoia/make_summary_from_splits.py --split_dir splits/sequoia/label_train_val_loop/train_p100
```

### 2. SSL pretraining

**MAE:**

```bash
python ssl_pretraining/mae/main_pretrain_imagenet_init.py \
  --data_path $DATA_ROOT/ssl_pretrain/chips_512_4ch/train/all \
  --output_dir $OUTPUT_ROOT/ssl/mae_vit_small
```

Convert checkpoint for downstream:

```bash
python ssl_pretraining/mae/convert_mae_to_timm_vit.py \
  --mae_checkpoint ... --output_path ...
```

**MoCo v3 (Swin Tiny):**

```bash
python ssl_pretraining/moco/train_moco_swin_tiny.py \
  --train_data $DATA_ROOT/ssl_pretrain/chips_512_4ch/train/all \
  --val_data $DATA_ROOT/ssl_pretrain/chips_512_4ch/val/all \
  --output_dir $OUTPUT_ROOT/ssl/moco_swin_tiny
```

### 3. Downstream training

Single run:

```bash
cd downstream/specdeepmap
python train.py \
  --input_folder ../../Sequoia_train_loop/100p \
  --pretrained_weights 4
```

Full experiment grid (splits × ImageNet / none / SSL × frozen / unfrozen):

```bash
# Sequoia dataset
python train_loop.py --dataset sequoia

# RedEdge dataset
python train_loop.py --dataset rededge

# DPT + ViT models
python train_loop_dpt_vit.py --dataset sequoia --vit_size small
python train_loop_dpt_vit.py --dataset rededge --vit_size base
```

See **[docs/downstream_sequoia_rededge.md](docs/downstream_sequoia_rededge.md)** for detailed usage.

## Using Pre-trained Checkpoints

Download checkpoints from **Zenodo** ([https://doi.org/10.5281/zenodo.21532723](https://doi.org/10.5281/zenodo.21532723)), then:

```bash
# Set environment variables to checkpoint locations
export SSL_MOCO_SWIN_CHECKPOINT=/path/to/moco_v3_swin_tiny_79ep.ckpt
export SSL_MOCO_VIT_CHECKPOINT=/path/to/moco_v3_vit_small_199ep.ckpt
export SSL_MAE_VIT_CHECKPOINT=/path/to/mae_vit_small_399ep.pth
export SSL_MAE_VIT_BASE_CHECKPOINT=/path/to/mae_vit_base_399ep.pth

# Then run downstream training with SSL weights
cd downstream/specdeepmap
python train_loop.py --dataset sequoia  # Uses MoCo Swin (weight index 4)
python train_loop_dpt_vit.py --dataset sequoia --vit_size small  # Uses MAE ViT
```

## License

SpecDeepMap components retain their original license — see `downstream/specdeepmap/LICENSE_specdeepmap.md`.
