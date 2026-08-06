# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import rasterio

import timm
try:
    import timm.optim.optim_factory as optim_factory
    # Check if add_weight_decay exists (deprecated in timm >= 1.0.10)
    if not hasattr(optim_factory, 'add_weight_decay'):
        optim_factory = None
except (ImportError, AttributeError):
    optim_factory = None

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae_mod_imagenet_init as models_mae

from engine_pretrain import train_one_epoch, validate_one_epoch


import random
print('cwd',os.getcwd())
print(torch.cuda.is_available())


class RandomCropTransform:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image):
        """Apply random crop to the image"""
        C, H, W = image.shape  # Assuming image has shape (C, H, W)
        top = random.randint(0, H - self.crop_size)
        left = random.randint(0, W - self.crop_size)
        return image[:, top:top + self.crop_size, left:left + self.crop_size]


# Use original torchvision transforms via v2 API which supports tensors
# v2 transforms work directly on tensors, so we use RandomCrop instead of RandomResizedCrop
def create_multispectral_transforms(input_size=224, mean=None, std=None):
    """Create torchvision transform pipeline for 4-channel multispectral tensors.
    
    Uses torchvision.transforms.v2 which supports tensors directly.
    Uses simple RandomCrop (no scaling/resizing) instead of RandomResizedCrop.
    """
    mean = mean if mean is not None else [0.161385, 0.146847, 0.312700, 0.352734] # new means for limited to 4cm

    std = std if std is not None else [0.103881, 0.130328, 0.145030, 0.168922] # new means for limited to 4cm

    
    # v2 transforms work on tensors (C, H, W) directly
    transform_train = transforms_v2.Compose([
        #transforms_v2.RandomCrop(size=(input_size, input_size)),
        transforms_v2.RandomResizedCrop(size=(input_size, input_size), scale=(0.2, 1.0),ratio=(1.0,1.0)), 
        transforms_v2.RandomHorizontalFlip(p=0.5),
        transforms_v2.RandomVerticalFlip(p=0.5),
        transforms_v2.Normalize(mean=mean, std=std),
    ])
    
    return transform_train


class MultiSpectralChipDataset(Dataset):
    """MultiSpectralChipDataset for MAE training.
    
    Loads 4-channel multispectral TIFF chips, applies initial crop, scales to [0,1],
    then applies torchvision-style transforms.
    """
    
    def __init__(
        self,
        root: str,
        crop_size: int = None,
        input_size: int = 224,
        mean: list = None,
        std: list = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            root: Root directory where the data is located
            crop_size: Size to crop from larger images (e.g., 512->224) before transforms
            input_size: Final output size after RandomResizedCrop (e.g., 224)
            mean: Normalization mean for 4 channels
            std: Normalization std for 4 channels
        """
        self.root = Path(root)
        self.crop_transform = RandomCropTransform(crop_size)
        self.transform = create_multispectral_transforms(input_size=input_size, mean=mean, std=std)
        self.chips = []

        # Find all chip files
        for chip_file in self.root.glob("*.tif"):
            self.chips.append(chip_file)

        if not self.chips:
            raise RuntimeError(f"No chip files found in {root}")

        print(f"Found {len(self.chips)} chips")

        # Verify the first chip has expected channels
        with rasterio.open(self.chips[0]) as src:
            n_channels = src.count
            print(f"Number of channels in first chip: {n_channels}")

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get a single sample from the dataset.

        Args:
            index: Index of the item to get

        Returns:
            Tensor of shape (C, H, W) - compatible with MAE
        """
        chip_path = self.chips[index]

        # Read the multi-spectral image
        with rasterio.open(chip_path) as src:
            image = src.read()  # Shape: (C, H, W), dtype typically uint8

        # Initial crop: 512x512 -> crop_size (e.g., 224x224)
        #image = self.crop_transform(image)
        
        # Convert to float32 and normalize to [0, 1] (scale from 8bit to 0-1)
        image = image.astype(np.float32) / 255.0

        # Convert to tensor
        image = torch.from_numpy(image)  # Shape: (C, H, W), range [0, 1]

        # Apply torchvision-style transforms (RandomCrop, RandomHorizontalFlip, Normalize)
        image = self.transform(image)

        return image  # Return tensor directly (not dict) for MAE compatibility - shape (C, input_size, input_size)
    
    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Length of the dataset
        """
        return len(self.chips)


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=4, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_small_patch16', type=str, metavar='MODEL',
                        help='Name of model to train (e.g., mae_vit_base_patch16, mae_vit_small_patch16)')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # ImageNet initialization parameters
    parser.add_argument('--init_from_imagenet', action='store_true',
                        help='Initialize encoder from ImageNet-pretrained ViT or DINO-pretrained model (default: True)')
    parser.add_argument('--no_init_from_imagenet', dest='init_from_imagenet', action='store_false',
                        help='Disable ImageNet initialization (random init instead)')
    parser.set_defaults(init_from_imagenet=True)
    parser.add_argument('--imagenet_model_name', default='vit_small_patch16_224.augreg_in21k', type=str,
                        help='timm model name for initialization (e.g., vit_base_patch16_224.augreg_in21k, vit_small_patch16_224.augreg_in21k)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--train_data', default='', type=str,
                        help='Directory containing unlabeled training .tif chips for SSL pretraining')
    parser.add_argument('--val_data', default='', type=str,
                        help='Directory containing unlabeled validation .tif chips (auto-detected if not provided)')

    parser.add_argument('--output_dir', default='',
                        help='Directory for checkpoints and logs')
    parser.add_argument('--log_dir', default='',
                        help='TensorBoard log directory (defaults to output_dir)')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint path (empty = train from scratch)')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    # Single-GPU / single-process training (no SLURM / no DDP needed)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # Use MultiSpectralChipDataset instead of ImageFolder
    # Pipeline: Load 512x512 -> RandomCrop to crop_size -> /255 to [0,1] -> 
    #           RandomCrop -> RandomHorizontalFlip -> Normalize
    dataset_train = MultiSpectralChipDataset(
        root=args.train_data,
        crop_size=448,  # Initial crop size (e.g., 224) from larger images (512x512)
        input_size=args.input_size,  # Final output size after RandomResizedCrop
        mean=[0.153467, 0.141859, 0.309777, 0.363888],
        std=[0.104461, 0.131514, 0.147038, 0.17462]
    )
    print(f"Dataset: {dataset_train}")
    print(f"Dataset size: {len(dataset_train)}")

    # simple random sampler for single-process training
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # Optional validation dataset
    data_loader_val = None
    if args.val_data and os.path.exists(args.val_data):
        dataset_val = MultiSpectralChipDataset(
            root=args.val_data,
            crop_size=448,
            input_size=args.input_size,
            mean=[0.153467, 0.141859, 0.309777, 0.363888],
            std=[0.104461, 0.131514, 0.147038, 0.17462]
        )
        print(f"Validation dataset: {dataset_val}")
        print(f"Validation size: {len(dataset_val)}")
        
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False,
        )

    # define the model - use 4 channels for multispectral (G, R, RE, NIR)
    # Optionally initialize from ImageNet if --init_from_imagenet is set
    model = models_mae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss, 
        in_chans=4,
        init_from_imagenet=args.init_from_imagenet,
        imagenet_model_name=args.imagenet_model_name
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    if args.init_from_imagenet:
        print(f"Encoder initialized from ImageNet-pretrained: {args.imagenet_model_name}")

    # Single-process training: world_size = 1
    eff_batch_size = args.batch_size * args.accum_iter
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    # Use timm's add_weight_decay if available, otherwise use manual implementation
    if optim_factory and hasattr(optim_factory, 'add_weight_decay'):
        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    else:
        # Fallback for newer timm versions where add_weight_decay is deprecated
        def add_weight_decay(model, weight_decay):
            """Group parameters: apply weight decay to weights, not to bias and norm layers."""
            decay = []
            no_decay = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith('.bias') or 'norm' in name.lower():
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': decay, 'weight_decay': weight_decay},
                {'params': no_decay, 'weight_decay': 0.0}
            ]
        param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        # Run validation if validation data is provided
        val_stats = {}
        if data_loader_val is not None:
            val_stats = validate_one_epoch(
                model, data_loader_val,
                device, epoch,
                log_writer=log_writer,
                args=args
            )
        
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,}

        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # Print epoch summary
        train_loss = train_stats.get('loss', 0.0)
        val_loss = val_stats.get('loss', 0.0) if val_stats else None
        if val_loss is not None:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        else:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from repo_paths import ssl_output_dir, ssl_pretrain_data  # noqa: E402

    args = get_args_parser()
    args = args.parse_args()
    if not args.train_data:
        args.train_data = str(ssl_pretrain_data() / "train" / "all")
    if not args.val_data:
        # Auto-detect validation data in sibling val/ folder
        val_path = ssl_pretrain_data() / "val" / "all"
        if val_path.exists():
            args.val_data = str(val_path)
    if not args.output_dir:
        args.output_dir = str(ssl_output_dir() / "mae_vit_small")
    if not args.log_dir:
        args.log_dir = args.output_dir
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

