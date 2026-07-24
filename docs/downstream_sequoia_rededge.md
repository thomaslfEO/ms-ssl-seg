# Downstream segmentation

Trainer code lives in `downstream/specdeepmap/` (standalone SpecDeepMap fork, no QGIS).

## Data Structure

The training scripts expect data organized in split folders with percentage-based naming:

```
Sequoia_train_loop/          or    Rededge_train_loop/
├── 5p/                             ├── 5p/
│   ├── images/                     │   ├── images/
│   ├── labels/                     │   ├── labels/
│   ├── train_files.csv             │   ├── train_files.csv
│   ├── validation_files.csv        │   ├── validation_files.csv
│   ├── Normalize_Bands.csv         │   ├── Normalize_Bands.csv
│   └── Summary_train_val.csv       │   └── Summary_train_val.csv
├── 10p/                            ├── 10p/
├── 25p/                            ├── 25p/
├── 50p/                            ├── 50p/
├── 75p/                            ├── 75p/
└── 100p/                           └── 100p/
```

Each split folder contains:
- **`train_files.csv`** & **`validation_files.csv`**: List of image/mask pairs
- **`Normalize_Bands.csv`**: Per-split band normalization (mean, std)
- **`Summary_train_val.csv`**: Per-split class statistics and weights
- **`images/`** & **`labels/`**: Image and label tiles

## Single training run

```bash
cd downstream/specdeepmap
python train.py \
  --input_folder "Sequoia_train_loop/100p" \
  --logdirpath "outputs/downstream/sequoia/logs" \
  --logdirpath_model "outputs/downstream/sequoia/checkpoints" \
  --pretrained_weights 4 \
  --backbone tu-swin_s3_tiny_224
```

## Experiment loops

| Script | Purpose |
|--------|---------|
| `train_loop.py` | Grid: splits × {ImageNet, none, SSL Swin} × {frozen, unfrozen} |
| `train_loop_dpt_vit.py` | Grid: DPT + ViT-small or ViT-base × {ImageNet, none, MAE SSL} × {frozen, unfrozen} |
| `train_loop_fastsiam.py` | Same grid, SSL weight 4 only (FastSiam ckpt) |
| `testing_loop.py` | Evaluate all train_loop checkpoints; save IoU list per split × weight × freeze |

### Dataset Selection

All loop scripts support both datasets via `--dataset` argument:

```bash
cd downstream/specdeepmap

# Sequoia (default)
python train_loop.py --dataset sequoia
python train_loop_dpt_vit.py --dataset sequoia --vit_size small

# RedEdge
python train_loop.py --dataset rededge
python train_loop_dpt_vit.py --dataset rededge --vit_size base
```

**Path resolution:**
- `--dataset sequoia` → uses `Sequoia_train_loop/` (or `$SEQUOIA_SPLITS_ROOT`)
- `--dataset rededge` → uses `Rededge_train_loop/` (or `$REDEDGE_SPLITS_ROOT`)
- Or specify custom path: `--splits_root /path/to/your/splits`

**Split discovery:**
- Legacy format: `train_p5`, `train_p10`, `train_p100`
- New format: `5p`, `10p`, `25p`, `50p`, `75p`, `100p`

Checkpoints are saved under `{splits_root}/checkpoints/{split}/{weight}_{frozen|unfrozen}/`.

**Normalization & class weights:**
- Each split automatically uses its own `Normalize_Bands.csv` and `Summary_train_val.csv`
- Enabled by default with `--normalization` and `--class_weights_balanced`

**DPT + ViT-Small** (set `SSL_MAE_VIT_CHECKPOINT` for MAE SSL runs):

```bash
python train_loop_dpt_vit.py --vit_size small --splits_root ../../splits/sequoia/label_train_val_loop
```

**DPT + ViT-Base** (set `SSL_MAE_VIT_BASE_CHECKPOINT` for MAE SSL runs):

```bash
python train_loop_dpt_vit.py --vit_size base --splits_root ../../splits/sequoia/label_train_val_loop
```

DPT checkpoints go to `{splits_root}/checkpoints_dpt_vit_small/` or `checkpoints_dpt_vit_base/`.

## Batch evaluation (all checkpoints)

Evaluate every trained checkpoint and save a flat IoU list:

```bash
# U-Net + Swin (train_loop.py)
python testing_loop.py --preset swin --splits_root ../../splits/sequoia/label_train_val_loop

# DPT + ViT-Small / ViT-Base
python testing_loop.py --preset dpt_vit_small --splits_root ../../splits/sequoia/label_train_val_loop
python testing_loop.py --preset dpt_vit_base --splits_root ../../splits/sequoia/label_train_val_loop
```

Outputs (JSON list + CSV table):

```text
{splits_root}/test_results/checkpoints/iou_results.json
{splits_root}/test_results/checkpoints/iou_results.csv
```

Each JSON entry contains: `split`, `weight_name`, `freeze`, `iou`, `accuracy`, `recall`, `precision`, `f1`, and the checkpoint path.

Per-experiment detailed class metrics are also written to `{experiment_dir}/test_results.csv`.

Use `--skip_existing` to reuse cached per-experiment CSVs, and `--test_data_csv` for one global test set.

## RedEdge

Use the same scripts with:

```bash
python train.py --input_folder "$DATA_ROOT/rededge/test_data_512x512_20min_percent"
```

Split CSVs are in `splits/rededge/test_data_512x512_20min_percent/`.

## Testing

Single checkpoint:

```bash
python test.py \
  --test_data_csv path/to/test_files.csv \
  --model_checkpoint path/to/model.ckpt \
  --csv_output path/to/iou.csv
```

All checkpoints from a training loop:

```bash
python testing_loop.py --preset swin --splits_root ../../splits/sequoia/label_train_val_loop
```

See `downstream/specdeepmap/STANDALONE_README.md` for full CLI reference.
