import torch
from torch.utils.data import random_split


def dataset_split(dataset, val_pct, test_pct=None):
    if test_pct is None:
        val_length = int(len(dataset) * val_pct)
        train_length = len(dataset) - val_length
        return random_split(dataset, [train_length, val_length])
    else:
        val_length = int(len(dataset) * val_pct)
        test_length = int(len(dataset) * test_pct)
        train_length = len(dataset) - (val_length + test_length)
        return random_split(
            dataset=dataset,
            lengths=[train_length, val_length, test_length],
            generator=torch.Generator().manual_seed(0),
        )
