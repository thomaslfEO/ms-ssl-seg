# SSL pretraining

## MAE (ViT)

Entry point: `ssl_pretraining/mae/main_pretrain_imagenet_init.py`

Trains a 4-channel masked autoencoder on unlabeled chips. Defaults resolve via `repo_paths.ssl_pretrain_data()` and `ssl_output_dir()` when `--data_path` / `--output_dir` are omitted.

After training, convert weights for timm/segmentation backbones:

- `convert_mae_to_timm_vit.py` — patch16
- `convert_mae_to_timm_vit_patch8.py` — patch8

Set `SSL_MAE_VIT_CHECKPOINT` for ViT/DPT downstream runs.

## MoCo v3

Canonical script: `ssl_pretraining/moco/train_moco_swin_tiny.py`

Set `SSL_MOCO_SWIN_CHECKPOINT` to the best `.ckpt` before running downstream with `--pretrained_weights 4`.

## Outputs

Checkpoints and TensorBoard logs are written to `$OUTPUT_ROOT/ssl/` by default (gitignored).
