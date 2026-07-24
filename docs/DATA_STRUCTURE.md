# Data Structure for Downstream Experiments

## Overview

The downstream segmentation experiments use a standardized folder structure with split-specific data configurations. This ensures each training split uses its own normalization parameters and class weights.

## Required Directory Structure

```
{Sequoia|Rededge}_train_loop/
├── 5p/
│   ├── images/                    # (or referenced in CSV)
│   ├── labels/                    # (or referenced in CSV)
│   ├── train_files.csv            # Required: image,mask pairs
│   ├── validation_files.csv       # Required: image,mask pairs
│   ├── Normalize_Bands.csv        # Required: per-band mean/std
│   └── Summary_train_val.csv      # Required: class weights & statistics
├── 10p/
├── 25p/
├── 50p/
├── 75p/
└── 100p/
```

## Required Files per Split

### 1. `train_files.csv` & `validation_files.csv`
```csv
image,mask
images/tile_001.tif,labels/tile_001.tif
images/tile_002.tif,labels/tile_002.tif
```

### 2. `Normalize_Bands.csv`
Per-split normalization parameters for each spectral band:
```csv
Band_Number,std,mean,std and mean already scaled by scaler
1,0.104461,0.153467,255
2,0.131514,0.141859,255
3,0.147038,0.309777,255
4,0.174622,0.363888,255
```

### 3. `Summary_train_val.csv`
Per-split class distribution and weights:
```csv
Class ID,Train Count,Train Percentage,Validation Count,Validation Percentage,Class Train Weight,Scaler,Ignored Background : Class Zero
1,28714747,97.0,3212137,97.0,0.03,None,No
2,818949,2.77,91586,2.77,0.9723,None,No
3,70144,0.24,7893,0.24,0.9976,None,No
```

## Location Resolution

Training scripts resolve data locations in this order:

1. **Command-line argument**: `--splits_root /path/to/splits`
2. **Dataset selection**: `--dataset {sequoia|rededge}`
   - Sequoia → `Sequoia_train_loop/` or `$SEQUOIA_SPLITS_ROOT`
   - RedEdge → `Rededge_train_loop/` or `$REDEDGE_SPLITS_ROOT`
3. **Environment variable**: `$DATA_ROOT/{dataset}/train_loop`
4. **Repository default**: `{repo_root}/{Dataset}_train_loop/`

## Usage Examples

### Training with Sequoia data
```bash
cd downstream/specdeepmap

# Use default Sequoia location
python train_loop.py --dataset sequoia

# Or specify custom path
python train_loop.py --splits_root /path/to/Sequoia_train_loop
```

### Training with RedEdge data
```bash
cd downstream/specdeepmap

# Use default RedEdge location
python train_loop.py --dataset rededge

# Or specify custom path
python train_loop.py --splits_root /path/to/Rededge_train_loop
```

### Testing all checkpoints
```bash
cd downstream/specdeepmap

# Evaluate Sequoia checkpoints
python testing_loop.py --dataset sequoia --preset swin

# Evaluate RedEdge checkpoints
python testing_loop.py --dataset rededge --preset dpt_vit_small
```

## Why Split-Specific Configs?

Each training split (5p, 10p, ..., 100p) has different:
- **Class distributions**: Smaller splits may have imbalanced classes
- **Normalization statistics**: Mean/std computed only on that split's training data
- **Class weights**: Adjusted based on that split's class balance

This ensures fair comparison across experiments and prevents data leakage.

## Current Locations

Based on your repository structure:
- Sequoia: `C:\IEEE_acess_code - Kopie\Sequoia_train_loop\`
- RedEdge: `C:\IEEE_acess_code - Kopie\Rededge_train_loop\`

Both contain splits: `5p`, `10p`, `25p`, `50p`, `75p`, `100p`
