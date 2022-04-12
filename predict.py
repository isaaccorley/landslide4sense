import argparse
import glob
import os

import h5py
from hydra.utils import instantiate
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.datasets import Landslide4SenseDataModule
from src.models import SegmentationModel


def write_mask(mask, path):
    f = h5py.File(path, "w")
    f.create_dataset(name="mask", shape=(128, 128), dtype="uint8", data=mask)
    f.close()


@torch.no_grad()
def main(log_dir, output_directory, device, split):
    pl.seed_everything(0, workers=True)
    os.makedirs(output_directory, exist_ok=True)

    # Load checkpoint and config
    conf = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    ckpt = glob.glob(os.path.join(log_dir, "checkpoints", "*.ckpt"))[0]

    # Load model
    model = SegmentationModel.load_from_checkpoint(ckpt)
    model = model.to(device)
    model.eval()

    # Load datamodule and dataloader
    datamodule = instantiate(conf.datamodule)
    datamodule.setup()

    if split == "val":
        dataloader = datamodule.predict_dataloader()

    # Predict
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            filenames = [filename.replace("image", "mask") for filename in batch["filename"]]
            filenames = [os.path.join(output_directory, filename) for filename in filenames]
            x = batch["image"].to(device)
            masks = model(x)
            masks = masks.softmax(dim=1).argmax(dim=1).cpu().numpy().astype("uint8")
            for filename, mask in zip(filenames, masks):
                write_mask(mask, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Path to log directory containing config.yaml and checkpoint",
    )
    parser.add_argument(
        "--predict_on",
        type=str,
        default="val",
        choices=["val"],
        help="Dataset to generate predictions of",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        required=True,
        help="Path to output_directory to save predictions",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    main(args.log_dir, args.output_directory, args.device, args.predict_on)