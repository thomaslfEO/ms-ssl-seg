# Data layout

All GeoTIFF chips stay **outside** the repository. The repo only ships CSV split manifests under `splits/`.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `DATA_ROOT` | Root for all local datasets (default: `./data`) |
| `OUTPUT_ROOT` | Checkpoints, logs (default: `./outputs`) |
| `SSL_DATA_PATH` | Unlabeled chips for SSL (default: `$DATA_ROOT/ssl_pretrain/chips_512_4ch`) |
| `SSL_OUTPUT_DIR` | SSL checkpoint output (default: `$OUTPUT_ROOT/ssl`) |
| `SSL_MOCO_SWIN_CHECKPOINT` | MoCo Swin ckpt for downstream weight index 4 |
| `SSL_FASTSIAM_CHECKPOINT` | FastSiam ckpt for FastSiam loop scripts |
| `SSL_MAE_VIT_CHECKPOINT` | MAE ViT ckpt for DPT/ViT experiments |
| `SEQUOIA_SPLITS_ROOT` | Override Sequoia split folder |
| `REDEDGE_SPLITS_ROOT` | Override RedEdge split folder |

## Expected folder structure

```text
$DATA_ROOT/
├── ssl_pretrain/chips_512_4ch/
│   ├── train/all/*.tif
│   └── val/all/*.tif
├── sequoia/
│   ├── filtered/
│   │   ├── images/
│   │   └── labels/
│   └── label_train_val_loop/
│       ├── train_p10/
│       │   ├── images/
│       │   ├── labels/
│       │   ├── train_files.csv
│       │   ├── validation_files.csv
│       │   ├── Summary_train_val.csv
│       │   └── Normalize_Bands.csv
│       └── train_p100/ ...
└── rededge/test_data_512x512_20min_percent/
    ├── images/
    ├── labels/
    └── *.csv
```

CSV columns: `image,mask` with paths relative to the split folder.

## Using repo manifests with local rasters

Option A — copy CSVs next to your rasters under `$DATA_ROOT` and set `SEQUOIA_SPLITS_ROOT`.

Option B — symlink:

```bash
ln -s /your/sequoia/chips $DATA_ROOT/sequoia/label_train_val_loop
```

Then copy or symlink `splits/sequoia/label_train_val_loop/train_p10/*.csv` into each split folder.
