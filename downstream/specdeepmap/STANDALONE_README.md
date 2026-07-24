# Standalone Deep Learning Trainer and Tester

This directory contains standalone Python scripts that allow you to train and test deep learning models **without QGIS or enmapbox dependencies**. These scripts can be run in a regular Python environment.

## Files

- `train.py` - Standalone training script
- `test.py` - Standalone testing script
- `simple_feedback.py` - Simple feedback class that replaces QGIS feedback
- `core_deep_learning_trainer_mica_swin.py` - Updated trainer core (QGIS dependencies removed)
- `core_tester_focal.py` - Updated tester core (QGIS dependencies removed)

## Requirements

Install the required Python packages:

```bash
pip install torch torchvision lightning segmentation-models-pytorch pandas numpy gdal torchmetrics
```

## Usage

### Training

Train a model using the `train.py` script:

```bash
python train.py \
    --input_folder /path/to/dataset/folder \
    --logdirpath /path/to/tensorboard/logs \
    --logdirpath_model /path/to/save/models \
    --arch 0 \
    --backbone tu-vit_small_patch16_224.augreg_in21k \
    --batch_size 8 \
    --n_epochs 50 \
    --lr 0.0003 \
    --device 1
```

#### Required Arguments:
- `--input_folder`: Path to folder containing train_files.csv, validation_files.csv, and Summary_train_val.csv
- `--logdirpath`: Path for saving Tensorboard logs
- `--logdirpath_model`: Path for saving model checkpoints

#### Optional Arguments:
- `--arch`: Architecture index (0=U-Net, 1=U-Net++, 2=DeepLabV3+, 3=SegFormer, 4=2D-Justo-UNet-Simple, 5=DPT, default: 0)
- `--backbone`: Backbone name (default: tu-vit_small_patch16_224.augreg_in21k)
- `--pretrained_weights`: Pretrained weights index (0=imagenet, 1=None, 2=Sentinel_2_TOA_Resnet18, 3=Sentinel_2_TOA_Resnet50, 4=MicaSense_SR_Swin_s3_tiny, default: 4)
- `--checkpoint`: Path to checkpoint file to continue training (optional)
- `--freeze_encoder`: Freeze backbone encoder (default: True)
- `--no_freeze_encoder`: Do not freeze backbone encoder
- `--data_aug`: Enable data augmentation (default: True)
- `--no_data_aug`: Disable data augmentation
- `--batch_size`: Batch size (default: 8)
- `--n_epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.0003)
- `--lr_finder`: Use automatic learning rate finder
- `--early_stop`: Enable early stopping
- `--class_weights_balanced`: Use balanced class weights (default: True)
- `--no_class_weights`: Do not use balanced class weights
- `--normalization`: Enable data normalization (default: True)
- `--no_normalization`: Disable data normalization
- `--device`: Device (0=cpu, 1=gpu, default: 1)
- `--num_workers`: Number of workers for data loading (default: 0)
- `--device_numbers`: Number of devices to use (default: 1)
- `--num_models`: Number of models to save (-1 means save each epoch, default: -1)

### Testing

Test a trained model using the `test.py` script:

```bash
python test.py \
    --test_data_csv /path/to/test_files.csv \
    --model_checkpoint /path/to/model.ckpt \
    --csv_output /path/to/output_iou.csv \
    --device 1 \
    --export_folder /path/to/export/predictions
```

#### Required Arguments:
- `--test_data_csv`: Path to test_files.csv (created by Dataset Maker)
- `--model_checkpoint`: Path to model checkpoint file (.ckpt)
- `--csv_output`: Path to output CSV file for IoU scores

#### Optional Arguments:
- `--device`: Device (0=cpu, 1=gpu, default: 1)
- `--no_data_label_mask`: Crop unclassified labels (0) from prediction before export (default: True)
- `--keep_no_data_labels`: Keep unclassified labels in exported predictions
- `--export_folder`: Folder to export prediction images as GeoTIFF (optional)

## Example Workflow

1. **Prepare your dataset** using the Dataset Maker (or manually create train_files.csv, validation_files.csv, test_files.csv, and Summary_train_val.csv)

2. **Train a model**:
   ```bash
   python train.py \
       --input_folder ./my_dataset \
       --logdirpath ./logs \
       --logdirpath_model ./models \
       --arch 0 \
       --backbone resnet18 \
       --batch_size 16 \
       --n_epochs 100 \
       --device 1
   ```

3. **Test the model**:
   ```bash
   python test.py \
       --test_data_csv ./my_dataset/test_files.csv \
       --model_checkpoint ./models/best_model.ckpt \
       --csv_output ./results/iou_scores.csv \
       --export_folder ./results/predictions \
       --device 1
   ```

## Changes Made

The following changes were made to enable standalone execution:

1. **Removed QGIS dependencies**: All `from qgis.*` imports have been removed
2. **Removed enmapbox dependencies**: Imports now use try/except to work with or without enmapbox
3. **Made feedback optional**: The `feedback` parameter is now optional and defaults to `None`
4. **Created SimpleFeedback class**: A simple feedback class replaces QgsProcessingFeedback for standalone use
5. **Updated core files**: 
   - `core_deep_learning_trainer_mica_swin.py` - U-Net + Swin training
   - `core_deep_learning_trainer_mica_vit.py` - DPT + ViT training
   - `core_tester_focal.py` - Standalone evaluation

## Notes

- The scripts will automatically use the best checkpoint based on validation IoU
- Progress is printed to console instead of QGIS feedback
- All functionality runs without QGIS/enmapbox (QGIS plugin wrappers were removed from this repo)
