"""
A series of helper functions.
"""

import torch
from torch.utils.data import dataset

def print_dataset_attributes(train_set: dataset, val_set: dataset):
    # Print total sample count
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")

