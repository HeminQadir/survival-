import os
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset.data_split_fold import datafold_read
from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    SpatialPadd,
    MapTransform,
    EnsureChannelFirstd,
    Rotate90d, 
    Flipd,
    Lambdad,
    ToTensord
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, UNet
from monai.networks.layers import Norm

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    DataLoader,
    Dataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

from typing import Dict, Any

class MultiplyImageLabeld(MapTransform):
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(data)
        data["image"] = data["image"] * data["label"]
        return data
    
import warnings
warnings.filterwarnings("ignore")
import random 
from torch.utils.data import Subset
import torch
from tensorboardX import SummaryWriter

def get_loader(train_files, val_files, args):
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            Rotate90d(keys=["image", "label"], spatial_axes=(0, 1)),
            Flipd(keys=["label"], spatial_axis=0),
            Lambdad(keys=["label"], func=lambda label: (label >= 1) * 1),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            CropForegroundd(keys=["image", "label"], source_key="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            
            # Multiply image with label
            MultiplyImageLabeld(keys=["image", "label"]),  # keys parameter is optional here
            #EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image"]),
        ]
    )
    
    val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        Rotate90d(keys=["image", "label"], spatial_axes=(0, 1)),
        Flipd(keys=["label"], spatial_axis=0),
        Lambdad(keys=["label"], func=lambda label: (label >= 1) * 1),
        ScaleIntensityRanged(keys=["image"],
                                a_min=-175,
                                a_max=250,
                                b_min=0.0,
                                b_max=1.0,
                                clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        CropForegroundd(keys=["image", "label"], source_key="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, .25, 2.0),
            mode=("bilinear", "nearest"),
        ),
        # Multiply image with label
        MultiplyImageLabeld(keys=["image", "label"]),  # keys parameter is optional here
        #EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96), mode='constant'),
        ToTensord(keys=["image"]),
    ]
)

    # Training subset dataloader 
    if args.cache_dataset:
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            #cache_num=24,    # this guy make it so slow 
            cache_rate=args.cache_rate,
            num_workers=args.num_workers,
        )
    else:
        train_ds = Dataset(data=train_files, 
                           transform=train_transforms)

    #train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0)
        #collate_fn=list_data_collate, 
        #sampler=train_sampler, 
        #pin_memory=True)
    

    # Validation subset dataloader
    if args.cache_dataset:
        val_ds = CacheDataset(
            data=val_files, 
            transform=train_transforms, #val_transforms, 
            #cache_num=6,
            cache_rate=args.cache_rate, 
            num_workers=int(args.num_workers/2)
        )
    else:
        val_ds = Dataset(
            data=val_files, 
            transform=train_transforms, #val_transforms
        )

    #val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        num_workers=0
    )
    
    # Get the length of the dataset
    dataset_length = len(train_ds)
    # Generate a random range of integers with the maximum being the length of train_dataset
    subset_indices = random.sample(range(dataset_length), int(dataset_length/4)) 
    print(f"number of subset from training set {len(subset_indices)}")
    subset = Subset(train_ds, subset_indices)
    # This is already cached 
    subset_loader = DataLoader(
        subset, 
        batch_size=args.batch_size,                
        shuffle=False, 
        num_workers=0
    )

    return train_loader, val_loader, subset_loader


def get_loader_inference(val_files, args):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], image_only=True),
            EnsureChannelFirstd(keys=["image", "label"]),
            Rotate90d(keys=["image", "label"], spatial_axes=(0, 1)),
            Flipd(keys=["label"], spatial_axis=0),
            Lambdad(keys=["label"], func=lambda label: (label >= 1) * 1),
            ScaleIntensityRanged(keys=["image"],
                                 a_min=args.a_min,
                                 a_max=args.a_max,
                                 b_min=args.b_min,
                                 b_max=args.b_max,
                                 clip=True,
            ),
            #CropForegroundd(keys=["image", "label"], source_key="image"),
            CropForegroundd(keys=["image", "label"], source_key="label"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            #EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            ToTensord(keys=["image"]),
        ]
    )

    # Validation subset dataloader
    if args.cache_dataset:
        val_ds = CacheDataset(
            data=val_files, 
            transform=val_transforms, 
            #cache_num=6,
            cache_rate=args.cache_rate, 
            num_workers=int(args.num_workers/2)
        )
    else:
        val_ds = Dataset(
            data=val_files, 
            transform=val_transforms
        )

    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        num_workers=0
    )

    return val_loader