# Data Structure

## Directory Structure

```
{Sequoia|Rededge}_train_loop/
├── 5p/
├── 10p/
├── 25p/
├── 50p/
├── 75p/
└── 100p/
```

Each split folder contains:
- `train_files.csv` - Training image/label pairs
- `validation_files.csv` - Validation image/label pairs  
- `Normalize_Bands.csv` - Per-split normalization statistics
- `Summary_train_val.csv` - Per-split class weights
- `images/`, `labels/` - Image tiles (optional if paths in CSV)

## Path Configuration

Use `--dataset sequoia` or `--dataset rededge` to automatically load the correct data directory.

Or set environment variables:
```bash
export SEQUOIA_SPLITS_ROOT=/path/to/Sequoia_train_loop
export REDEDGE_SPLITS_ROOT=/path/to/Rededge_train_loop
```

## Usage

```bash
cd downstream/specdeepmap

# Sequoia dataset
python train_loop.py --dataset sequoia

# RedEdge dataset
python train_loop.py --dataset rededge

# Or specify custom path
python train_loop.py --splits_root /path/to/your/data
```

See [downstream_sequoia_rededge.md](downstream_sequoia_rededge.md) for detailed usage.
