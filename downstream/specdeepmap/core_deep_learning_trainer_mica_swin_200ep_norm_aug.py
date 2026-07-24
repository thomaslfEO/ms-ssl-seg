import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from repo_paths import ssl_moco_swin_checkpoint  # noqa: E402


def _require_moco_swin_checkpoint() -> Path:
    path = ssl_moco_swin_checkpoint()
    if path is None:
        raise ValueError(
            "Set SSL_MOCO_SWIN_CHECKPOINT when using MicaSense_SR_Swin_s3_tiny weights."
        )
    if not path.is_file():
        raise FileNotFoundError(f"SSL MoCo Swin checkpoint not found: {path}")
    return path

# import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

# Import GDAL - handle Windows DLL path issues
try:
    from osgeo import gdal
except ImportError as e:
    # On Windows, sometimes need to add GDAL DLLs to PATH
    import os
    import sys
    # Try to find GDAL in conda environment
    conda_env = os.environ.get('CONDA_PREFIX', '')
    if conda_env:
        gdal_dll_path = os.path.join(conda_env, 'Library', 'bin')
        if os.path.exists(gdal_dll_path) and gdal_dll_path not in os.environ.get('PATH', ''):
            os.environ['PATH'] = gdal_dll_path + os.pathsep + os.environ.get('PATH', '')
    # Try importing again
    from osgeo import gdal

from torch.utils.data import Dataset
from torchmetrics import JaccardIndex
from torchvision import transforms
from torchvision.transforms import v2

### new
#from focal_loss.focal_loss import FocalLoss

from segmentation_models_pytorch.losses import FocalLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
# GDAL already imported above


class FocalLoss(nn.Module):
    """
    Simple FocalLoss - copy this into your code.
    No dependencies, just works.
    """

    def __init__(self, alpha=None, gamma=2.0, ignore_index=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
            elif isinstance(alpha, torch.Tensor):
                self.register_buffer('alpha', alpha.float())
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # inputs: (batch, num_classes, H, W)
        # targets: (batch, H, W)

        # Only pass ignore_index if it's not None
        if self.ignore_index is not None:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                # Move alpha to same device as targets
                alpha = self.alpha.to(targets.device)
                alpha_t = alpha[targets]
                if self.ignore_index is not None:
                    alpha_t = torch.where(targets == self.ignore_index,
                                          torch.zeros_like(alpha_t), alpha_t)
                focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
            else:
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Handle ignore_index in reduction
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)
            focal_loss = focal_loss * valid_mask.float()

            if self.reduction == 'mean':
                valid_count = valid_mask.float().sum()
                return focal_loss.sum() / valid_count if valid_count > 0 else torch.tensor(0.0,
                                                                                           device=focal_loss.device,
                                                                                           requires_grad=True)
        elif self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum() if self.reduction == 'sum' else focal_loss


try:
    from enmapbox.apps.SpecDeepMap.utils_resnet import ResNet18_Weights, ResNet50_Weights
except ImportError:
    from utils_resnet import ResNet18_Weights, ResNet50_Weights

# Data augmentation

transforms_v2 = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
])

# preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')


from torchvision.models._api import WeightsEnum


# Simple Model Unet
# source https://github.com/NTNU-SmallSat-Lab/s_l_c_segm_hyp_img/blob/main/Justoetal_models_public_released.py

class JustoUNetSimple(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(JustoUNetSimple, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(input_channels, 6, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(6)
        self.enc_conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(12)

        # Decoder
        self.dec_conv1 = nn.Conv2d(12, 6, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(6)
        self.dec_conv2 = nn.Conv2d(6, num_classes, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(num_classes)

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc_bn1(self.enc_conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.enc_bn2(self.enc_conv2(x)))
        x = F.max_pool2d(x, 2)

        # Decoder
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.relu(self.dec_bn1(self.dec_conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.dec_conv2(x)
        x = self.dec_bn2(x)
        return F.softmax(x, dim=1)


def model_2D_Justo_UNet_Simple(input_channels, num_classes):
    # Assuming input_size is (H, W, C)
    model = JustoUNetSimple(input_channels, num_classes)
    return model


_model_weights = {
    "Sentinel_2_TOA_Resnet18": [ResNet18_Weights.SENTINEL2_ALL_MOCO],
    "Sentinel_2_TOA_Resnet50": [ResNet50_Weights.SENTINEL2_ALL_MOCO],
    "LANDSAT_TM_TOA_Resnet18": [ResNet18_Weights.LANDSAT_TM_TOA_MOCO],
    "LANDSAT_ETM_TOA_Resnet18": [ResNet18_Weights.LANDSAT_ETM_TOA_MOCO],
    "LANDSAT_OLI_TIRS_TOA_Resnet18": [ResNet18_Weights.LANDSAT_OLI_TIRS_TOA_MOCO],
    "LANDSAT_ETM_SR_Resnet18": [ResNet18_Weights.LANDSAT_ETM_SR_MOCO],
    "LANDSAT_OLI_SR_Resnet18": [ResNet18_Weights.LANDSAT_OLI_SR_MOCO],
}


def get_weight(name: str) -> WeightsEnum:

    if name is None:
        return None

    for weight_name, weight_enum in _model_weights.items():
        if isinstance(weight_name, str):
            for sub_weight_enum in weight_enum:
                if name == str(sub_weight_enum):
                    return sub_weight_enum

    raise ValueError(f'{name} is not a valid WeightsEnum')


def preprocessing_imagenet():
    # Read the CSV into a pandas DataFrame

    # ImageNet BGR mean values (reversed RGB values)
    imagenet_bgr_mean = [0.406, 0.456, 0.485]  # BGR order
    imagenet_bgr_stds = [0.229, 0.224, 0.225]

    # Create and return the PyTorch normalization transform
    return transforms.Compose([
        # transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=imagenet_bgr_mean, std=imagenet_bgr_stds)
        # Normalize using the modified means and stds
    ])


def preprocessing_imagenet_additional(csv_path):
    # Read the CSV into a pandas DataFrame
    data = pd.read_csv(csv_path)

    # Extract the 'mean' column from the CSV, assuming it has a 'mean' column
    all_means = data['mean'].tolist()

    # ImageNet BGR mean values (reversed RGB values)
    imagenet_bgr_mean = [0.406, 0.456, 0.485]  # BGR order
    imagenet_bgr_stds = [0.229, 0.224, 0.225]

    # Replace the first 3 channels with ImageNet BGR mean
    all_means[:3] = imagenet_bgr_mean

    # Extract standard deviation column, if available, else default to 1
    all_stds = data['std'].tolist()
    all_stds[:3] = imagenet_bgr_stds

    # Create and return the PyTorch normalization transform
    return transforms.Compose([
        # transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=all_means, std=all_stds)  # Normalize using the modified means and stds
    ])


def preprocessing_sentinel2_TOA():
    """
    Sentinel-2 Top-of-Atmosphere reflectance normalization.
    All channels are scaled between 0 and 10000, no specific normalization used.
    """
    return transforms.Compose([
        # transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=0, std=10000)  # Normalize by dividing by 10000 (range 0-10000)
    ])


def preprocessing_normalization_csv(csv_path):
    # Read the CSV into a pandas DataFrame
    data = pd.read_csv(csv_path)

    all_means = data['mean'].tolist()
    all_stds = data['std'].tolist()

    # Create and return the PyTorch normalization transform
    return transforms.Compose([
        # transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=all_means, std=all_stds)  # Normalize using the modified means and stds
    ])


def get_preprocessing_pipeline(pretrained_weights, channels, normalization, normalization_path):
    if pretrained_weights == 'imagenet' and channels == 3:
        preprocessing = preprocessing_imagenet()
        print('preprocessing_imagenet')
    elif pretrained_weights == 'imagenet' and channels > 3:
        assert normalization_path != None, "Normalization CSV must be computed to use imagenet for more then 3 channel to harmonize preprocessing."
        preprocessing = preprocessing_imagenet_additional(normalization_path)
        print('preprocessing_imagenet_more channels')
    elif pretrained_weights == 'Sentinel_2_TOA_Resnet18' or pretrained_weights == 'Sentinel_2_TOA_Resnet50':
        preprocessing = preprocessing_sentinel2_TOA()
        print('preprocessing_sentinel')  # Sentinel-2 normalization for additional channels
    elif pretrained_weights is None and normalization == True and normalization_path != None:
        preprocessing = preprocessing_normalization_csv(normalization_path)
        print('preprocessing_normalization')
    else:
        preprocessing = None  # No preprocessing if conditions don't match

    return preprocessing


class CustomDataset(Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks.
    """

    def __init__(
            self,
            csv_paths_dataframe: pd.DataFrame,
            transform: Optional = None,
            num_classes: Optional[int] = None,
            preprocess_input: Optional = None,
            remove: Optional = None,
            scaler_loader: Optional = None,
            remap: Optional = None,
            # Use A.Compose for transforms
    ):
        """

        Args:
            x_paths (pd.DataFrame): a dataframe with a row for each chip. There must be a column for chip_id,
                and a column with raster image, and a column with the corresponding mask.

            transforms : Compose object for image augmentations.
        """
        self.data = csv_paths_dataframe
        # Remove the extra comma, and use the actual DataFrame
        self.transform = transform
        self.num_classes = num_classes
        self.preprocess_input = preprocess_input
        self.remove = remove
        self.scaler_loader = scaler_loader
        self.remap = remap
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):

        # id = self.data.loc[idx]
        img_path = self.data.loc[idx, 'image']
        mask_path = self.data.loc[idx, 'mask']

        data = gdal.Open(img_path)
        mask = gdal.Open(mask_path)
        1

        # channel first
        data_array = data.ReadAsArray().astype(np.float32)
        data_array = data_array                        ############################# scale adjusted fix to 255
        mask_array = mask.ReadAsArray().astype(np.float32)

        #close gdal ###################################################### this needs to go in trainer so doesn't get stuck.
        if data is not None:
            data = None
        if mask is not None:
            mask = None

        # remap using dict
        forward_array = mask_array.copy()
        for old, new in self.remap.items():
            forward_array[mask_array == old] = new

        del mask_array

        mask_array = torch.as_tensor(forward_array, dtype=torch.int64)

        if self.transform != None:
            mask_array = np.array(mask_array)
            data_array = np.array(data_array)

            data_array, mask_array = self.transform(data_array, mask_array)

        else:
            data_array = torch.as_tensor(data_array, dtype=torch.float32)
            mask_array = torch.as_tensor(mask_array, dtype=torch.float32)

        if self.scaler_loader != None:
            data_array /= self.scaler_loader

        if self.preprocess_input != None:
            data_array = torch.as_tensor(data_array, dtype=torch.float32)
            mask_array = torch.as_tensor(mask_array, dtype=torch.float32)
            data_array = self.preprocess_input(data_array)

        item = {'image': data_array, 'mask': mask_array}
        return item


class MyModel(L.LightningModule):
    def __init__(
            self,
            # bands: List[str],
            train_data: Optional[pd.DataFrame] = None,
            # y_train: Optional[pd.DataFrame] = None,
            val_data: Optional[pd.DataFrame] = None,
            # y_val: Optional[pd.DataFrame] = None,
            hparams: dict = None,
            feedback = None
    ):
        """

        Args:

            hparams (dict, optional): Dictionary of additional modeling parameters.
        """
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()

        # optional modeling params
        self.architecture = self.hparams.get("architecture", 'Unet')
        self.backbone = self.hparams.get("backbone", 'resnet18')
        self.weights = self.hparams.get("weights", None)
        self.learning_rate = self.hparams.get("lr", None)
        self.num_workers = self.hparams.get("num_workers", 0)
        self.batch_size = self.hparams.get("batch_size", None)
        self.acc = self.hparams.get("acc", 'gpu')
        self.transform = self.hparams.get("transform")
        self.in_channels = self.hparams.get("in_channels")
        self.classes = self.hparams.get("classes")
        self.class_weights = self.hparams.get("class_weights")
        self.checkpoint_path = self.hparams.get("checkpoint_path")
        self.freeze_encoder = self.hparams.get("freeze_backbone")
        self.img_x = self.hparams.get("img_x")
        self.img_y = self.hparams.get("img_y")
        self.preprocess = self.hparams.get("preprocess", None)
        self.counter = 0
        self.remove_b = self.hparams.get("remove_background_class")
        self.scaler = self.hparams.get("scaler")
        self.class_values = self.hparams.get("class_values")
        self.forward_mapping = self.hparams.get("forward_mapping")
        self.reverse_mapping = self.hparams.get("reverse_mapping")
        self.n_epochs = self.hparams.get("epochs")

        if self.classes == 1:
            # self.iou = JaccardIndex(task="binary",num_classes=self.classes, ignore_index=self.ignore_index)
            # self.val_iou = JaccardIndex(task="binary",num_classes=self.classes, ignore_index=self.ignore_index)
            self.iou = JaccardIndex(task="binary", num_classes=self.classes)
            self.val_iou = JaccardIndex(task="binary", num_classes=self.classes)


        elif self.classes > 1 and self.remove_b == 'Yes':
            self.iou = JaccardIndex(task="multiclass", num_classes=self.classes, ignore_index=0)
            self.val_iou = JaccardIndex(task="multiclass", num_classes=self.classes, ignore_index=0)
        else:
            self.iou = JaccardIndex(task="multiclass", num_classes=self.classes)
            self.val_iou = JaccardIndex(task="multiclass", num_classes=self.classes)

        # Instantiate datasets, model, and trainer params if provided

        self.train_dataset = CustomDataset(
            csv_paths_dataframe=train_data,
            transform=self.transform,
            num_classes=self.classes,  #
            preprocess_input=self.preprocess,
            remove=self.remove_b,
            scaler_loader=self.scaler,
            remap=self.forward_mapping

        )

        self.val_dataset = CustomDataset(
            csv_paths_dataframe=val_data,
            transform=None,
            num_classes=self.classes,
            preprocess_input=self.preprocess,
            remove=self.remove_b,
            scaler_loader=self.scaler,
            remap=self.forward_mapping
        )

        self.model = self._prepare_model()

    def forward(self, image: torch.Tensor):
        # Forward pass
        return self.model(image)

    def training_step(self, batch: dict, batch_idx: int):
        """
        Training step.

        Args:

        """
        if self.train_dataset.data is None:
            raise ValueError(
                "Train Dataset must be specified to train model"
            )

        # Switch on training mode
        self.model.train()
        torch.set_grad_enabled(True)

        # Load images and labels

        x = batch["image"]
        y = batch["mask"].long()
        # non_zero_mask = batch["zero_mask"]

        if self.acc == 'gpu':
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        preds = self.forward(x)  # Model predictions

        if self.remove_b == 'Yes':

            train_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction="mean", ignore_index=0)(preds,
                                                                                                                y)

        else:
            train_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction="mean")(preds, y)


        #focal loss test

        # Convert class weights to list
        #classweight_list = self.class_weights.cpu().tolist() if isinstance(self.class_weights, torch.Tensor) else list(
         #   self.class_weights)
        # In your validation_step:
        #focal_loss_fn = FocalLoss(
         #   alpha=[0.02, 0.25, 0.7],
          #  gamma=2.0,  # Higher = more focus on rare class
           # ignore_index=0 if self.remove_b == 'Yes' else None,
            #reduction='mean'
        #)

        #train_loss = focal_loss_fn(preds, y)
        # Simple FocalLoss - no ignore_index


        # Mask the predictions for IoU computation

        preds = torch.argmax(preds, dim=1)

        # Compute IoU only on non-zero masked values
        train_iou = self.iou(preds, y)

        self.log_dict({'train_loss': train_loss, 'train_iou': train_iou}
                      , on_step=True, on_epoch=True, prog_bar=True, logger=True
                      )
        # Accessing step-level and epoch-level metrics during training

        return {'loss': train_loss, 'train_iou': train_iou}

    def validation_step(self, batch: dict, batch_idx: int):
        """
        Validation step.

        Args:

        """
        if self.val_dataset.data is None:
            raise ValueError(
                "Validation Datset must be specified to train model"
            )

        # Switch on validation mode
        self.model.eval()
        torch.set_grad_enabled(False)

        # Load images and labels

        x = batch["image"]
        y = batch["mask"].long()  # Ground truth mask
        # non_zero_mask = batch["zero_mask"]

        if self.acc == 'gpu':
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        preds = self.forward(x)  # Model predictions

        if self.remove_b == 'Yes':

            val_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction="mean", ignore_index=0)(
                preds, y)

        else:
            val_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights, reduction="mean")(preds, y)


        ### test focal loss

        # Convert class weights to list
        #classweight_list = self.class_weights.cpu().tolist() if isinstance(self.class_weights, torch.Tensor) else list(
         #   self.class_weights)

        # Simple FocalLoss - no ignore_index

        #focal_loss_fn = FocalLoss(
            #alpha=[0.02, 0.97, 0.99],
           # gamma=3.0,  # Higher = more focus on rare class
          #  ignore_index=0 if self.remove_b == 'Yes' else None,
         #   reduction='mean'
        #)

        #val_loss = focal_loss_fn(preds, y)


        preds = torch.argmax(preds, dim=1)

        # Compute IoU using the validation metric tracker (self.val_iou)
        # This ensures proper epoch-level aggregation for checkpoint monitoring
        val_iou = self.val_iou(preds, y)

        # Log metrics - Lightning automatically creates 'val_iou_epoch' from 'val_iou' when on_epoch=True
        # We also explicitly log 'val_iou_epoch' in on_validation_epoch_end to ensure it's available
        self.log_dict({'val_loss': val_loss, 'val_iou': val_iou}
                      , on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=False
                      )

        return {'val_loss': val_loss, 'val_iou': val_iou}  # val_iou

    def predict(self, image: torch.Tensor):

        self.model.eval()
        torch.set_grad_enabled(False)

        if self.scaler != None:
            image /= self.scaler

        if self.preprocess != None:
            image = torch.as_tensor(image)

            image = self.preprocess(image)
        else:
            image = torch.as_tensor(image)

        logits = self(image)

        pred1 = torch.softmax(logits, dim=1)

        pred2 = torch.argmax(pred1, dim=1)

        pred2 = pred2.squeeze()

        pred2= pred2.cpu().numpy()
        #
        reverse_array = pred2.copy()


        for old, new in self.reverse_mapping.items():
            reverse_array[pred2 == old] = new

        # Cleanup memory
        del pred2

        return reverse_array

    def train_dataloader(self):
        # DataLoader class for training
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True if self.acc == 'gpu' else False,
            drop_last=True,
            persistent_workers=False,  # Prevent worker process issues with GDAL
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def val_dataloader(self):
        # DataLoader class for validation
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True if self.acc == 'gpu' else False,
            drop_last=True,
            persistent_workers=False,  # Prevent worker process issues with GDAL
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.n_epochs, eta_min=1e-6)
        return [opt], [sch]

    def on_train_epoch_end(self):
        self.counter += 1  # Increment the counter
        self.log('counter', self.counter)
    
    def on_validation_epoch_end(self):
        """Compute and log epoch-level IoU metric for checkpoint monitoring."""
        # Compute the epoch-level IoU from accumulated validation batches
        val_iou_epoch = self.val_iou.compute()
        # Log it explicitly - this is what ModelCheckpoint monitors
        # CRITICAL: Must log with on_epoch=True and ensure it's available to callbacks
        # Use .item() to convert tensor to scalar for logging
        val_iou_value = val_iou_epoch.item() if hasattr(val_iou_epoch, 'item') else float(val_iou_epoch)
        self.log('val_iou_epoch', val_iou_value, 
                 prog_bar=True, 
                 logger=True, 
                 sync_dist=False,
                 on_epoch=True, 
                 on_step=False)
        print(f"[OK] Logged val_iou_epoch={val_iou_value:.4f} for ModelCheckpoint")
        # Reset metric for next epoch
        self.val_iou.reset()
        # Also reset training IoU metric
        self.iou.reset()

    def _prepare_model(self):


        # overwrite wrong backbone if pretrained weights
        if self.weights == 'Sentinel_2_TOA_Resnet18':
            self.backbone = 'resnet18'
        elif self.weights == 'Sentinel_2_TOA_Resnet50':
            self.backbone = 'resnet50'

        # build arch.

        if self.architecture == 'Unet':
            model = smp.Unet(
                encoder_name=self.backbone,
                encoder_weights='imagenet' if self.weights == 'imagenet' else None,
                in_channels=self.in_channels,
                classes=self.classes
            )
        elif self.architecture == 'Unet++':
            model = smp.UnetPlusPlus(
                encoder_name=self.backbone,
                encoder_weights='imagenet' if self.weights == 'imagenet' else None,
                in_channels=self.in_channels,
                classes=self.classes
            )
        elif self.architecture == 'DeepLabV3+':
            model = smp.DeepLabV3Plus(
                encoder_name=self.backbone,
                encoder_weights='imagenet' if self.weights == 'imagenet' else None,
                in_channels=self.in_channels,
                classes=self.classes
            )

        elif self.architecture == 'SegFormer':
            model = smp.Segformer(
                encoder_name=self.backbone,
                encoder_weights='imagenet' if self.weights == 'imagenet' else None,
                in_channels=self.in_channels,
                classes=self.classes
            )

        elif self.architecture == 'DPT':
            model = smp.DPT(
                encoder_name=self.backbone,
                encoder_weights='imagenet' if self.weights == 'imagenet' else None,
                in_channels=self.in_channels,
                classes=self.classes
            )

        elif self.architecture == 'JustoUNetSimple':

            model = model_2D_Justo_UNet_Simple(input_channels=self.in_channels, num_classes=self.classes)

        # loader for backbone

        if self.weights not in ['imagenet', None]:

            if self.weights == 'Sentinel_2_TOA_Resnet18':
                assert self.in_channels == 13, f'Input channels should be equal to 13 , but is {self.in_channels}'
                weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
                #self.backbone = 'resnet18'
                state_dict = weights.get_state_dict(progress=True)
                model.encoder.load_state_dict(state_dict)

            elif self.weights == 'Sentinel_2_TOA_Resnet50':
                assert self.in_channels == 13, f'Input channels should be equal to 13 , but is {self.in_channels}'
                weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
                #self.backbone = 'resnet50'
                state_dict = weights.get_state_dict(progress=True)
                model.encoder.load_state_dict(state_dict)

            elif self.weights == "MicaSense_SR_Swin_s3_tiny":
                #assert self.in_channels == 7, f'Input channels should be equal to 7 , but is {self.in_channels}'

                #self.backbone = 'tu-swin_s3_tiny_224'
                #self.backbone = 'resnet18'
                
                # Load MAE pretrained ViT encoder for DPT architecture
                if self.architecture == 'DPT':
                    #self.backbone = 'tu-vit_small_patch16_224.augreg_in21k'

                    # used this checkpoint for all small experiments
                    #checkpoint_path = r"/home/thomaslf/icarus_data_zenodo/MAE/checkpoints_mae_vit_small_patch16/vit_small_patch16_224_encoder_399ep.pth"

                    #mae vit small imagenet ini 399 ep
                    #checkpoint_path = '/home/thomaslf/icarus_data_zenodo/MAE/checkpoints_mae_vit_small_patch16_imagenet_true/vit_small_patch16_224_mae_pretrained_imagenet_ini_399ep.pth'


                    # used this checkpoint for all base experiments , not coverted jet
                    #self.backbone ='tu-vit_base_patch16_224.augreg_in21k'
                    #checkpoint_path = '/home/thomaslf/icarus_data_zenodo/MAE/checkpoints_mae_vit_base_patch16_imagenet_true/checkpoint-399.pth'

                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                    encoder_state_dict = checkpoint['model']
                    
                    # DPT uses 'model.' prefix for encoder parameters
                    smp_encoder_dict = {}
                    for key, value in encoder_state_dict.items():
                        smp_key = f'model.{key}'
                        smp_encoder_dict[smp_key] = value
                    
                    missing, unexpected = model.encoder.load_state_dict(smp_encoder_dict, strict=False)
                    print(f"Loaded MAE pretrained ViT encoder: {len(smp_encoder_dict) - len(missing)}/{len(smp_encoder_dict)} parameters")
                    if missing:
                        print(f"Missing keys: {len(missing)}")
                    if unexpected:
                        print(f"Unexpected keys: {len(unexpected)}")
                else:
                    #self.backbone = 'tu-swin_s3_tiny_224'
                    # assert self.in_channels == 7, f'Input channels should be equal to 7 , but is {self.in_channels}'

                    self.backbone = 'tu-swin_s3_tiny_224'

                    #79 epoch mocov3
                    #path = 'C:/specdeepmap_without_env/SpecDeepMap/moco_v3_swin_s3_epoch=79_train_train_loss_epoch=0.0887_val_val_loss_epoch=0.1229.ckpt'
                    

                    #200 ep  mocov3
                    path = str(_require_moco_swin_checkpoint())
                    
                    #path = '/home/thomaslf/icarus_data_zenodo/512_ALL_4channel_moco_v3_swin_tiny_imagenet_normalized_t_007/moco_v3_swin_s3_epoch=79_train_train_loss_epoch=0.0887_val_val_loss_epoch=0.1229.ckpt'
                    #149
                    #path = '/home/thomaslf/icarus_data_zenodo/512_ALL_4channel_moco_v3_swin_tiny_imagenet_normalized_t_007/moco_v3_swin_s3_epoch=149_train_train_loss_epoch=0.1450_val_val_loss_epoch=0.1703.ckpt'

                    # swin norm 128 - this is absolute garbage performs super bad compared to others already on 5p train
                    #path = '/home/thomaslf/icarus_data_zenodo/512_ALL_4channel_moco_v2_swin_tiny_imagenet_128_outputdim_normalized/moco_v2_swin_tiny_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=1.8616_val_val_loss_epoch=1.7397.ckpt'

                    #self.backbone = 'resnet18'

                    # moco v2 - 256x256 + 128 dim + normalisation: worse then unfrozen on 5 p train

                    #path ='/home/thomaslf/icarus_data_zenodo/512_ALL_4channel_moco_v2_resnet18_imagenet_128_outputdim_normalised_crop256x256/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=1.4234_val_val_loss_epoch=1.5932.ckpt'







                    # path = "C:/MA_LT_no_data255__mix_max_gdal_red_edge/moco_v2_resnet50_epoch_epoch=49_train_train_loss_epoch=3.7240_val_val_loss_epoch=3.9379.ckpt"

                    # path = "C:/MA_LT_no_data255__mix_max_gdal_red_edge/moco_v2_resnet50_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=2.5619_val_val_loss_epoch=2.6850.ckpt"

                    # 1. non-mean std agrinir moco , dim 256  - also crap
                    #path = "C:/MA_LT_no_data255__mix_max_gdal_red_edge/Rededge_WEEDMAP/Moco_pretrained/res18_moco_AGrnirnir/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=0.3858_val_val_loss_epoch=0.4425.ckpt"

                    #  2. non-mean std sentienl2 moco, dim 256

                    # path = "C:/MA_LT_no_data255__mix_max_gdal_red_edge/Rededge_WEEDMAP/Moco_pretrained/resnet_moco__sentinel/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=0.2701_val_val_loss_epoch=0.3181.ckpt"

                    #  3. non mean std moco- imagenet non mean std dim 256

                    # path ="C:/MA_LT_no_data255__mix_max_gdal_red_edge/Rededge_WEEDMAP/Moco_pretrained/res18_moco_imagnet/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=0.7750_val_val_loss_epoch=0.9593.ckpt"

                    #  4. non meant std moco-none dim 256

                    # from here dim 128 , with mean

                    # 1 Imagenet

                    # path = "C:/MA_LT_no_data255__mix_max_gdal_red_edge/pretrained_mono_normalised_128/imagenet/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=0.8238_val_val_loss_epoch=0.9856.ckpt"

                    #

                    # Agrivison

                    # SAM
                    # self.backbone = 'tu-sam2_hiera_tiny'
                    # path = '/home/thomaslf/icarus_data_zenodo/512_ALL_4channel_moco_v2_swin_tiny_imagenet_128_outputdim_normalized/moco_v2_sam2_hiera_tiny_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=1.8616_val_val_loss_epoch=1.7397.ckpt'

                    # resnet imagenet moco-  256 din normalised , unfrozen slightly better then unforzen scratch
                    #self.backbone = 'resnet18'
                    #path ='/home/thomaslf/icarus_data_zenodo/512_ALL_4channel_moco_v2_resnet18_imagenet_256_outputdim_normalized/moco_v2_resnet18_ALL_4channel_epoch_epoch=189_train_train_loss_epoch=0.7698_val_val_loss_epoch=0.8022.ckpt'

                    # resnet imagenet init 128 output dim normalised
                    #path = '/home/thomaslf/icarus_data_zenodo/512_ALL_4channel_moco_v2_resnet18_imagenet_128_outputdim_normalized/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=0.8238_val_val_loss_epoch=0.9856.ckpt'

                    # resnet sentinel2 dim 128 normalised
                    #path = '/home/thomaslf/icarus_data_zenodo/512_ALL_4channel_moco_v2_resnet18_sentinel2_128_outputdim_normalized/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=0.3133_val_val_loss_epoch=0.3702.ckpt'
                    # agrivision normalized 128 dim

                    # path ='/home/thomaslf/icarus_data_zenodo/512_ALL_4channel_moco_v2_resnet18_agrivision_128_outputdim_normalized/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=0.4102_val_val_loss_epoch=0.3821.ckpt'

                    # resnet50
                    # path ='/home/thomaslf/icarus_data_zenodo/512_ALL_4channel_moco_v2_resnet50/moco_v2_resnet50_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=2.5619_val_val_loss_epoch=2.6850.ckpt'



                    checkpoint = torch.load(path, map_location=torch.device('cpu'))
                    print("Checkpoint hyper_parameters:", checkpoint.get('hyper_parameters', 'Not found'))

                    state_dict_mod = checkpoint['state_dict']
                    print("Original checkpoint keys (first 10):", list(state_dict_mod.keys())[:10])

                    # Get only the backbone keys (not backbone_momentum)
                    state_dict_mod = OrderedDict(
                        {k: v for k, v in state_dict_mod.items() if
                         k.startswith('backbone.') and not k.startswith('backbone_momentum')})
                    print(f"Filtered backbone keys count: {len(state_dict_mod)}")

                    # Remove the 'backbone.' prefix to match the target model's keys
                    state_dict_mod = OrderedDict(
                        {k.replace('backbone.', 'model.'): v for k, v in state_dict_mod.items()}
                    )

                    # Apply regex substitution: replace any character after 'layers' with '_'
                    # This handles cases like 'layers.0' -> 'layers_0'
                    state_dict_mod = OrderedDict(
                        {re.sub(r'(?<=layers).', '_', k): v for k, v in state_dict_mod.items()}
                    )

                    print("Keys after transformation (first 10):", list(state_dict_mod.keys())[:10])

                    # load whole model with weights
                    ignore_keys = {"model.norm.weight", "model.norm.bias"}

                    # Filter out unwanted keys
                    state_dict_mod = {k: v for k, v in state_dict_mod.items() if k not in ignore_keys}

                    # Get model encoder keys for comparison
                    model_encoder_keys = set(model.encoder.state_dict().keys())
                    checkpoint_keys = set(state_dict_mod.keys())

                    missing_keys = model_encoder_keys - checkpoint_keys
                    unexpected_keys = checkpoint_keys - model_encoder_keys

                    print(f"Model encoder expects {len(model_encoder_keys)} keys")
                    print(f"Checkpoint provides {len(checkpoint_keys)} keys")
                    if missing_keys:
                        print(f"Missing keys ({len(missing_keys)}): {list(missing_keys)[:10]}")
                    if unexpected_keys:
                        print(f"Unexpected keys ({len(unexpected_keys)}): {list(unexpected_keys)[:10]}")

                    # Try loading with strict=False to allow partial loading
                    try:
                        result = model.encoder.load_state_dict(state_dict_mod, strict=False)
                        if result.missing_keys:
                            print(f"Warning: {len(result.missing_keys)} keys were missing during loading")
                        if result.unexpected_keys:
                            print(f"Warning: {len(result.unexpected_keys)} keys were unexpected during loading")
                        print("Successfully loaded checkpoint weights into encoder")
                    except Exception as e:
                        print(f"Error loading state dict: {e}")
                        print("Attempting to load with only matching keys...")
                        # Filter to only include keys that exist in the model
                        matching_keys = {k: v for k, v in state_dict_mod.items() if k in model_encoder_keys}
                        model.encoder.load_state_dict(matching_keys, strict=False)
                        print(f"Loaded {len(matching_keys)} matching keys out of {len(state_dict_mod)} checkpoint keys")


                #path = "C:/MA_LT_no_data255__mix_max_gdal_red_edge/moco_v2_resnet50_epoch_epoch=49_train_train_loss_epoch=3.7240_val_val_loss_epoch=3.9379.ckpt"

                #path = "C:/MA_LT_no_data255__mix_max_gdal_red_edge/moco_v2_resnet50_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=2.5619_val_val_loss_epoch=2.6850.ckpt"

                # 1. non-mean std agrinir moco , dim 256
                #path = "C:/MA_LT_no_data255__mix_max_gdal_red_edge/Rededge_WEEDMAP/Moco_pretrained/res18_moco_AGrnirnir/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=0.3858_val_val_loss_epoch=0.4425.ckpt"

                #  2. non-mean std sentienl2 moco, dim 256

                #path = "C:/MA_LT_no_data255__mix_max_gdal_red_edge/Rededge_WEEDMAP/Moco_pretrained/resnet_moco__sentinel/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=0.2701_val_val_loss_epoch=0.3181.ckpt"

                #  3. non mean std moco- imagenet non mean std dim 256

                #path ="C:/MA_LT_no_data255__mix_max_gdal_red_edge/Rededge_WEEDMAP/Moco_pretrained/res18_moco_imagnet/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=0.7750_val_val_loss_epoch=0.9593.ckpt"

                #  4. non meant std moco-none dim 256


                # from here dim 128 , with mean

                # 1 Imagenet

                #path = "C:/MA_LT_no_data255__mix_max_gdal_red_edge/pretrained_mono_normalised_128/imagenet/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_train_loss_epoch=0.8238_val_val_loss_epoch=0.9856.ckpt"

                # Sentinel

                #Agrivison

                # None

                # moco v2 crop 256x256 imagenet init , mean-std norm
                #path = "C:/MA_LT_no_data255__mix_max_gdal_red_edge/Rededge_WEEDMAP/moco_quality_check/moco_v2_resnet18_ALL_4channel_epoch_epoch=199_train_crop_256x256.ckpt"

                # moco scale 0.7-1.0 no crop mean std

        if self.freeze_encoder == True:
            # Freeze encoder weights
            for param in model.encoder.parameters():
                param.requires_grad = False
            # Freeze BatchNorm layers
            for module in model.encoder.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.weight.requires_grad = False
                    module.bias.requires_grad = False

        return model


class CheckpointCleanupCallback(L.Callback):
    """Custom callback to clean up old checkpoints when save_top_k is set."""
    def __init__(self, logdirpath_model, save_top_k):
        super().__init__()
        self.logdirpath_model = logdirpath_model
        self.save_top_k = save_top_k
        self.saved_checkpoints = []  # List of (score, filepath) tuples
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Clean up old checkpoints after validation."""
        if self.save_top_k <= 0:
            return  # Don't clean up if saving all
            
        import os
        import glob
        import re
        
        # Get all checkpoint files
        checkpoint_pattern = os.path.join(self.logdirpath_model, '*.ckpt')
        all_checkpoints = glob.glob(checkpoint_pattern)
        
        if len(all_checkpoints) == 0:
            return  # No checkpoints yet
            
        # Get metric value for current epoch
        current_score = trainer.callback_metrics.get('val_iou_epoch')
        if current_score is None:
            print("WARNING: val_iou_epoch not found in callback_metrics, cannot clean up checkpoints")
            return  # Can't determine which to keep
            
        current_score = current_score.item() if hasattr(current_score, 'item') else float(current_score)
        epoch = trainer.current_epoch
        
        # Find the checkpoint file for current epoch by parsing filenames
        current_checkpoint = None
        for ckpt in all_checkpoints:
            # Check for epoch in filename (format: epoch=00001 or epoch=1)
            epoch_match = re.search(r'epoch[=_](\d+)', os.path.basename(ckpt))
            if epoch_match:
                ckpt_epoch = int(epoch_match.group(1))
                if ckpt_epoch == epoch:
                    current_checkpoint = ckpt
                    break
        
        # If we found the current checkpoint, add it to our list
        if current_checkpoint:
            # Remove it from list if it already exists (update score)
            self.saved_checkpoints = [(s, f) for s, f in self.saved_checkpoints if f != current_checkpoint]
            self.saved_checkpoints.append((current_score, current_checkpoint))
            print(f"Tracked checkpoint: epoch {epoch}, score {current_score:.4f}, file: {os.path.basename(current_checkpoint)}")
        else:
            # If we can't find current checkpoint, try to build list from all checkpoints
            print(f"Could not find checkpoint for epoch {epoch}, scanning all checkpoints...")
            for ckpt in all_checkpoints:
                # Try to extract score from filename
                score_match = re.search(r'val_iou_([\d.]+)', os.path.basename(ckpt))
                if score_match:
                    score = float(score_match.group(1))
                    if (score, ckpt) not in self.saved_checkpoints:
                        self.saved_checkpoints.append((score, ckpt))
        
        # Sort by score (descending for max mode)
        self.saved_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Keep only top k - delete the rest
        if len(self.saved_checkpoints) > self.save_top_k:
            to_delete = self.saved_checkpoints[self.save_top_k:]
            for score, filepath in to_delete:
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        print(f"[OK] Deleted old checkpoint: {os.path.basename(filepath)} (score: {score:.4f})")
                    except Exception as e:
                        print(f"[X] Failed to delete checkpoint {filepath}: {e}")
            # Update list to keep only top k
            self.saved_checkpoints = self.saved_checkpoints[:self.save_top_k]
            print(f"Kept top {self.save_top_k} checkpoints. Best score: {self.saved_checkpoints[0][0]:.4f}")
        else:
            print(f"Keeping {len(self.saved_checkpoints)} checkpoint(s) (within limit of {self.save_top_k})")


class FeedbackCallback(L.Callback):
    def __init__(self, feedback):
        super().__init__()
        self.feedback = feedback

    def on_train_batch_end(self,trainer, *args, **kwargs):
        # Check for cancellation after every batch
        if self.feedback and self.feedback.isCanceled():
            trainer.should_stop = True
            if self.feedback:
                self.feedback.pushInfo("TRAINING CANCELED BY USER !!!")
            else:
                print("TRAINING CANCELED BY USER !!!")
            raise KeyboardInterrupt("TRAINING CANCELED BY USER !!!")

            #raise KeyboardInterrupt("Training canceled by user.")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        max_epochs = trainer.max_epochs

        train_loss = trainer.callback_metrics.get('train_loss')
        train_iou = trainer.callback_metrics.get('train_iou')
        val_loss = trainer.callback_metrics.get('val_loss')
        val_iou = trainer.callback_metrics.get('val_iou')

        log_message = (
            f'Epoch {epoch }/{max_epochs-1} - '
            f'Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}'
        )

        if self.feedback:
            self.feedback.setProgress((epoch+1)/ max_epochs * 100)
            self.feedback.pushInfo(log_message)

            # Check if the user canceled the process
            #if self.feedback.isCanceled():
             #   trainer.should_stop = True#
              #  raise KeyboardInterrupt("Training canceled by user.")

        print(log_message)


def dl_train(
        input_folder,
        arch_index, backbone='resnet18', pretrained_weights_index=0,
        checkpoint_path=None,
        freeze_encoder=True, data_aug=True, batch_size=16, n_epochs=100, lr=0.0001, early_stop=True,
        class_weights_balanced=True,
        normalization_bool=True,
        num_workers=0, num_models=1, acc_type_index=None, acc_type_numbers=1, logdirpath_model=None,
        logdirpath='./logs', tune=True, feedback=None):

    arch_index_options = ['Unet', 'Unet++', 'DeepLabV3+', 'SegFormer', 'JustoUNetSimple','DPT']
    arch = arch_index_options[arch_index]

    pretrained_weights_options = ['imagenet', None, 'Sentinel_2_TOA_Resnet18',
                                  'Sentinel_2_TOA_Resnet50',"MicaSense_SR_Swin_s3_tiny"]                   #  ,'LANDSAT_TM_TOA_Resnet18','LANDSAT_ETM_TOA_Resnet18','LANDSAT_OLI_TIRS_TOA_Resnet18','LANDSAT_ETM_SR_Resnet18','LANDSAT_OLI_SR_Resnet18']
    pretrained_weights = pretrained_weights_options[pretrained_weights_index]

    if pretrained_weights == 'Sentinel_2_TOA_Resnet18':
        backbone = 'resnet18'

    elif pretrained_weights == 'Sentinel_2_TOA_Resnet50':
        backbone = 'resnet50'

    if arch == 'JustoUNetSimple':
        freeze_encoder = False

    # load data

    def fix_path(path):
        return path.replace('\\', '/')

    # Make sure the folder path uses forward slashes
    folder_path = fix_path(input_folder)

    train_data_path = folder_path + '/train_files.csv'
    val_data_path = folder_path + '/validation_files.csv'
    summary_data_path = folder_path + '/Summary_train_val.csv'

    train_data = pd.read_csv(train_data_path)
    for col in ["image", "mask"]:
        train_data[col] = train_data[col].apply(lambda rel_path: str(folder_path / Path(rel_path)))

    val_data = pd.read_csv(val_data_path)
    for col in ["image", "mask"]:
        val_data[col] = val_data[col].apply(lambda rel_path: str(folder_path / Path(rel_path)))

    print(val_data.head())

    summary_data = pd.read_csv(summary_data_path)

    # read from csv
    remove_zero_class = summary_data['Ignored Background : Class Zero'].tolist()[0]

    # create extra no-data class layer if yes

    # dynamic remapping of labeled data (handles uncontinious data labels , ignores 0 in class_values, important for iou calc in mapper/tester)
    original_values = sorted(summary_data['Class ID'].unique().tolist())
    n_classes = len(original_values)

    if remove_zero_class == 'Yes':
        original_values = [0] + original_values
        n_classes = len(original_values)
    cls_values = original_values

    forward_mapping = {original: idx for idx, original in enumerate(original_values)}

    # Create reverse mapping (new indices -> original)
    reverse_mapping = {idx: original for original, idx in forward_mapping.items()}
    print('forward mapping',forward_mapping)
    print('backward mapping',reverse_mapping)

    if 0 in cls_values:
        cls_values.remove(0)

    #  until here remapping look up tables for train and prediction

    scaler_list = summary_data['Scaler'].tolist()
    scaler = scaler_list[0]

    ignore_scaler_list = ([
        'Sentinel_2_TOA_Resnet18', 'Sentinel_2_TOA_Resnet50'])

    # 'LANDSAT_TM_TOA_Resnet18', 'LANDSAT_ETM_TOA_Resnet18','LANDSAT_OLI_TIRS_TOA_Resnet18', 'LANDSAT_ETM_SR_Resnet18', 'LANDSAT_OLI_SR_Resnet18'])

    scaler_value = None if pretrained_weights in ignore_scaler_list else summary_data['Scaler'].iloc[0]

    # Handle NaN values
    if isinstance(scaler_value, float) and math.isnan(scaler_value):
        scaler_value = None

    print(f"Scaler after handling NaN: {scaler} (type: {type(scaler)})")

    # data aug:
    if data_aug == True:
        # Assuming transform setup here
        transform = v2.Compose([
            v2.RandomRotation(degrees=45),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
    else:
        transform = None

    acc_type_options = ['cpu', 'gpu']
    acc_type = acc_type_options[acc_type_index]

    # balanced training #
    if class_weights_balanced == True:

        # Extract the 'weights' column as a list
        if remove_zero_class == 'Yes':
            weights_list = [0] + summary_data['Class Train Weight'].tolist()
        else:
            weights_list = summary_data['Class Train Weight'].tolist()

        if acc_type == 'gpu':

            weights_tensor = torch.as_tensor(weights_list, dtype=torch.float32).cuda()

        elif acc_type == 'cpu':
            weights_tensor = torch.as_tensor(weights_list, dtype=torch.float32)

    else:
        weights_tensor = None

    # Load the first image and mask to determine their dimensions
    first_image_path = train_data['image'].iloc[0]
    first_img = gdal.Open(first_image_path, gdal.GA_ReadOnly)
    in_channels = first_img.RasterCount
    first_img_x = first_img.RasterXSize
    first_img_y = first_img.RasterYSize

    if arch == 'JustoUNetSimple':
        backbone = None

    # initalize preprocessing in regards to normalization or pretrained weights
    if normalization_bool == True:
        normalize_data_path = folder_path + '/Normalize_Bands.csv'
        preprocess_input = get_preprocessing_pipeline(pretrained_weights, channels=in_channels,
                                                      normalization=normalization_bool,
                                                      normalization_path=normalize_data_path)
    else:
        preprocess_input = get_preprocessing_pipeline(pretrained_weights, channels=in_channels,
                                                      normalization=None,
                                                      normalization_path=None)

    model = MyModel(
        train_data=train_data,
        val_data=val_data,
        hparams={
            'in_channels': in_channels,
            'architecture': arch,
            'classes': n_classes,
            'batch_size': batch_size,
            'backbone': backbone,
            'weights': pretrained_weights,
            'epochs': n_epochs,
            'transform': transform,
            'lr': lr,
            'num_workers': num_workers,
            'acc': acc_type,
            'freeze_backbone': freeze_encoder,
            "class_weights": weights_tensor,
            'checkpoint_path': None,
            "img_x": first_img_x,
            "img_y": first_img_y,
            "preprocess": preprocess_input,
            "remove_background_class": remove_zero_class,
            "scaler": scaler_value,
            "forward_mapping": forward_mapping,
            "reverse_mapping": reverse_mapping,
            "class_values": cls_values

        }
        # feedback = feedback
    )

    if checkpoint_path == True:
        print('loaded from checkpoint')
        model = MyModel.load_from_checkpoint(checkpoint_path, train_data=train_data, val_data=val_data,
                                             hparams={'in_channels': in_channels,
                                                      'architecture': arch,
                                                      'classes': n_classes,
                                                      'batch_size': batch_size,
                                                      'backbone': backbone,
                                                      'weights': pretrained_weights,
                                                      'epochs': n_epochs,
                                                      'transform': transform,
                                                      'lr': lr,
                                                      'num_workers': num_workers,
                                                      'acc': acc_type,
                                                      'freeze_backbone': freeze_encoder,
                                                      "class_weights": weights_tensor,
                                                      'checkpoint_path': None,
                                                      "img_x": first_img_x,
                                                      "img_y": first_img_y,
                                                      "preprocess": preprocess_input,
                                                      "remove_background_class": remove_zero_class,
                                                      "scaler": scaler_value,
                                                      "forward_mapping": forward_mapping,
                                                      "reverse_mapping": reverse_mapping,
                                                      "class_values": cls_values},
                                             map_location=acc_type
                                             )

    # Callbacks
    if early_stop == True:
        early_stopping_callback = EarlyStopping("val_iou", mode="max", verbose=True, patience=20)

        # Configure save_top_k: -1 saves all, 1 saves only best, N saves top N
        # PyTorch Lightning: save_top_k=-1 means save all, save_top_k=1 means save only best
        # IMPORTANT: If num_models=-1, we want to save all, so save_top_k=-1
        # If num_models=1, we want to save only best, so save_top_k=1
        save_top_k = num_models  # Direct mapping: 1=save best, -1=save all, N=save top N
        print(f"ModelCheckpoint configuration: save_top_k={save_top_k}, num_models={num_models}")
        if save_top_k == -1:
            print("  WARNING: save_top_k=-1 will save ALL checkpoints (every epoch)")
        elif save_top_k == 1:
            print("  -> Should save ONLY the best model (highest val_iou_epoch)")
            print("  -> If all epochs are being saved, the metric 'val_iou_epoch' may not be found")
        
        # Create ModelCheckpoint - ensure metric name matches what we log
        # The metric 'val_iou_epoch' is logged in on_validation_epoch_end
        # CRITICAL: save_top_k=1 means save only the best model, -1 means save all
        # If the metric is not found, ModelCheckpoint may save all checkpoints
        checkpoint_callback = ModelCheckpoint(
            dirpath=logdirpath_model,
            monitor='val_iou_epoch',  # Must match the metric name logged in on_validation_epoch_end
            mode='max',  # Save model with highest val_iou_epoch
            filename='{epoch:05d}-val_iou_{val_iou_epoch:.4f}', 
            save_top_k=save_top_k,  # 1 = save only best, -1 = save all, N = save top N
            auto_insert_metric_name=False,
            save_last=False,  # Don't save last checkpoint separately
            every_n_epochs=1,  # Check every epoch
            save_on_train_epoch_end=False,  # CRITICAL: Only save after validation (when metric is available)
            enable_version_counter=False,
            verbose=True,
            save_weights_only=False
        )
        
        print(f"ModelCheckpoint configured: save_top_k={checkpoint_callback.save_top_k}, monitor='{checkpoint_callback.monitor}', mode='{checkpoint_callback.mode}'")
        if save_top_k == 1:
            print(f"  -> Will save ONLY the best model (highest val_iou_epoch)")
            print(f"  -> WARNING: If you see all epochs being saved, the metric 'val_iou_epoch' may not be available to ModelCheckpoint")
        elif save_top_k > 1:
            print(f"  -> Will save the top {save_top_k} models")
        else:
            print(f"  -> Will save all models (save_top_k={save_top_k})")

        feedback_callback = FeedbackCallback(feedback=feedback)
        
        callbacks_list = [checkpoint_callback, early_stopping_callback, feedback_callback]

        logger = TensorBoardLogger(save_dir=logdirpath, name="lightning_logs")
        trainer = L.Trainer(
            max_epochs=n_epochs,
            accelerator=acc_type,
            devices=acc_type_numbers,  # leads to gui crash, also starts multiprocess
            logger=logger,
            log_every_n_steps=1,
            callbacks=callbacks_list,
        )

        if tune == True:
            tuner = Tuner(trainer)

            # Run learning rate finder
            lr_finder = tuner.lr_find(model)

            # Results can be found in
            print('lr', lr_finder.results)

            new_lr = lr_finder.suggestion()
            # update hparams of the model
            model.hparams.lr = new_lr

            if feedback:
                feedback.pushInfo(f"learning rate finder suggested and used: {new_lr} as learning rate for training")
            else:
                print(f"learning rate finder suggested and used: {new_lr} as learning rate for training")

        trainer.fit(model)

    else:
        # Configure save_top_k: -1 saves all, 1 saves only best, N saves top N
        # PyTorch Lightning: save_top_k=-1 means save all, save_top_k=1 means save only best
        # IMPORTANT: If num_models=-1, we want to save all, so save_top_k=-1
        # If num_models=1, we want to save only best, so save_top_k=1
        save_top_k = num_models  # Direct mapping: 1=save best, -1=save all, N=save top N
        print(f"ModelCheckpoint configuration: save_top_k={save_top_k}, num_models={num_models}")
        if save_top_k == -1:
            print("  WARNING: save_top_k=-1 will save ALL checkpoints (every epoch)")
        elif save_top_k == 1:
            print("  -> Should save ONLY the best model (highest val_iou_epoch)")
            print("  -> If all epochs are being saved, the metric 'val_iou_epoch' may not be found")
        
        # Create ModelCheckpoint - ensure metric name matches what we log
        # The metric 'val_iou_epoch' is logged in on_validation_epoch_end
        # CRITICAL: save_top_k=1 means save only the best model, -1 means save all
        # If the metric is not found, ModelCheckpoint may save all checkpoints
        checkpoint_callback = ModelCheckpoint(
            dirpath=logdirpath_model,
            monitor='val_iou_epoch',  # Must match the metric name logged in on_validation_epoch_end
            mode='max',  # Save model with highest val_iou_epoch
            filename='{epoch:05d}-val_iou_{val_iou_epoch:.4f}', 
            save_top_k=save_top_k,  # 1 = save only best, -1 = save all, N = save top N
            auto_insert_metric_name=False,
            save_last=False,  # Don't save last checkpoint separately
            every_n_epochs=1,  # Check every epoch
            save_on_train_epoch_end=False,  # CRITICAL: Only save after validation (when metric is available)
            enable_version_counter=False,
            verbose=True,
            save_weights_only=False
        )
        
        print(f"ModelCheckpoint configured: save_top_k={checkpoint_callback.save_top_k}, monitor='{checkpoint_callback.monitor}', mode='{checkpoint_callback.mode}'")
        if save_top_k == 1:
            print(f"  -> Will save ONLY the best model (highest val_iou_epoch)")
            print(f"  -> WARNING: If you see all epochs being saved, the metric 'val_iou_epoch' may not be available to ModelCheckpoint")
        elif save_top_k > 1:
            print(f"  -> Will save the top {save_top_k} models")
        else:
            print(f"  -> Will save all models (save_top_k={save_top_k})")

        feedback_callback = FeedbackCallback(feedback=feedback)
        
        callbacks_list = [checkpoint_callback, feedback_callback]

        logdir = logdirpath
        logger = TensorBoardLogger(save_dir=logdir, name="lightning_logs")
        trainer = L.Trainer(
            max_epochs=n_epochs,
            accelerator=acc_type,
            devices=acc_type_numbers,  # leads to gui crash, also starts multiprocess
            logger=logger,
            log_every_n_steps=1,
            callbacks=callbacks_list,
        )

        if tune == True:
            tuner = Tuner(trainer)

            # Run learning rate finder
            lr_finder = tuner.lr_find(model)

            # Results can be found in
            print('lr', lr_finder.results)

            new_lr = lr_finder.suggestion()
            # update hparams of the model
            model.hparams.lr = new_lr

            if feedback:
                feedback.pushInfo(f"learning rate finder suggested and used: {new_lr} as learning rate for training")
            else:
                print(f"learning rate finder suggested and used: {new_lr} as learning rate for training")

        trainer.fit(model)

    return model, acc_type
