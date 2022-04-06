import argparse
import os
import shutil

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import OmegaConf

import src


def main(cfg_path, cfg):
    pl.seed_everything(0, workers=True)
    module = instantiate(cfg.module)
    augmentations = instantiate(cfg.augmentations)
    datamodule = instantiate(cfg.datamodule, augmentations=augmentations)
    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(model=module, datamodule=datamodule)
    shutil.copyfile(cfg_path, os.path.join(trainer.logger.log_dir, "config.yaml"))
    trainer.test(datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to config.yaml file"
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    main(args.cfg, cfg)
