import glob
import os

import h5py
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import draw_segmentation_masks

from src.transforms import transforms
from src.utils import dataset_split


class LandSlide4Sense(Dataset):
    folders = {"train": "TrainData", "val": "ValidData", "test": "TestData"}

    def __init__(self, root="data", split="train", transforms=None):
        assert split in self.folders
        self.root = os.path.join(root, self.folders[split])
        self.transforms = transforms
        self.images = sorted(glob.glob(os.path.join(self.root, "img", "*h5")))

        if os.path.exists(os.path.join(self.root, "mask")):
            self.masks = sorted(glob.glob(os.path.join(self.root, "mask", "*h5")))
        else:
            self.masks = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        image = self._load_image(path)
        sample = {"image": image, "filename": os.path.basename(path)}

        if self.masks is not None:
            path = self.masks[idx]
            mask = self._load_target(path)
            sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path):
        with h5py.File(path, "r") as f:
            x = f["img"][:]
            x = torch.from_numpy(x).to(torch.float)
            x = x.permute(2, 0, 1)
        return x

    def _load_target(self, path):
        with h5py.File(path, "r") as f:
            x = f["mask"][:]
            x = torch.from_numpy(x).to(torch.long)
        return x


class Landslide4SenseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        batch_size=2,
        num_workers=0,
        num_prefetch=2,
        val_pct=0.1,
        augmentations=nn.Identity(),
    ) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_prefetch = num_prefetch
        self.val_pct = val_pct
        self.transforms = transforms()
        self.augmentations = augmentations

    def preprocess(self, sample):
        sample["image"] = self.transforms(sample["image"])
        return sample

    def setup(self, stage=None):
        transforms = T.Compose([self.preprocess])
        dataset = LandSlide4Sense(self.root, split="train", transforms=transforms)
        self.train_dataset, self.val_dataset = dataset_split(
            dataset, val_pct=self.val_pct
        )
        self.predict_dataset = LandSlide4Sense(
            self.root, split="val", transforms=transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.num_prefetch,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.num_prefetch,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.num_prefetch,
            shuffle=False,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            batch["mask"] = rearrange(batch["mask"], "b h w -> b () h w")
            batch["mask"] = batch["mask"].to(torch.float)
            batch["image"], batch["mask"] = self.augmentations(
                batch["image"], batch["mask"]
            )
            batch["mask"] = batch["mask"].to(torch.long)
            batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")
        return batch

    def plot(self, y):
        x = torch.zeros(3, *y.shape).to(torch.uint8)
        y = y.cpu().to(torch.bool)
        image = draw_segmentation_masks(x, y, alpha=0.5, colors="red")
        return image
