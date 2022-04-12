from typing import Sequence

import kornia.augmentation as K
import torch.nn as nn
import torchvision.transforms as T

MEAN = (
    -0.4914,
    -0.3074,
    -0.1277,
    -0.0625,
    0.0439,
    0.0803,
    0.0644,
    0.0802,
    0.3000,
    0.4082,
    0.0823,
    0.0516,
    0.3338,
    0.7819,
)
STD = (
    0.9325,
    0.8775,
    0.8860,
    0.8869,
    0.8857,
    0.8418,
    0.8354,
    0.8491,
    0.9061,
    1.6072,
    0.8848,
    0.9232,
    0.9018,
    1.2913,
)


def augmentations() -> K.AugmentationSequential:
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomAffine(degrees=(0, 90), p=0.5),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2),
        data_keys=["input", "mask"],
    )


def rgb_augmentations() -> nn.Sequential:
    return nn.Sequential(
        K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.25)
    )


def transforms(
    mean: Sequence[float] = MEAN, std: Sequence[float] = STD
) -> T.Compose:
    return T.Compose([T.Normalize(mean=mean, std=std)])
