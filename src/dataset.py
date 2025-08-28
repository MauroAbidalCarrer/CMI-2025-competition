from os.path import join
from functools import partial
from tqdm.notebook import tqdm
from collections import defaultdict
from typing import Optional, Literal
from itertools import pairwise, starmap, product

import torch
import kagglehub 
import numpy as np
import pandas as pd
from numpy import ndarray
from torch import nn, Tensor
from torch.optim import Optimizer
from pandas import DataFrame as DF
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader as DL
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold

from config import *
from utils import seed_everything

class CMIDataset(TensorDataset):
    def __init__(self, device: torch.device):
        x = np.load(join("preprocessed_dataset", "X.npy")).swapaxes(1, 2)
        y = np.load(join("preprocessed_dataset", "Y.npy"))
        auxiliary_orientation_y = np.load(join("preprocessed_dataset", "orientation_Y.npy"))
        binary_demographics_y = np.load(join("preprocessed_dataset", "binary_demographics_Y.npy"))
        regression_demographics_y = np.load(join("preprocessed_dataset", "regres_demographics_Y.npy"))
        super().__init__(
            torch.from_numpy(x).to(device),
            torch.from_numpy(y).to(device),
            torch.from_numpy(auxiliary_orientation_y).to(device),
            torch.from_numpy(binary_demographics_y).to(device),
            torch.from_numpy(regression_demographics_y).to(device),
        )

    def __getitem__(self, index):
        return *super().__getitem__(index), index

def stratified_group_train_test_split(
    n_samples: int,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.2,
    # random_state: int = 42,
    ) -> tuple[ndarray, ndarray]:
    assert n_samples == len(y) == len(groups), \
        "dataset, y, and groups must have the same length"

    # Ensure test_size matches 1/k fractions (StratifiedGroupKFold requirement)
    n_splits = int(1 / test_size)
    if abs(test_size - (1 / n_splits)) > 1e-6:
        raise ValueError(f"test_size={test_size} is not supported exactly; "
                         f"use fractions like 0.5, 0.33, 0.25, 0.2, etc.")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True)

    return next(cv.split(np.zeros(n_samples), y, groups))


def split_dataset() -> tuple[Subset, Subset, Subset]:
    """Returns: experts training idx, gating model training idx, validation idx"""
    seed_everything(SEED)
    full_dataset = CMIDataset(torch.device("cpu"))
    seq_meta = pd.read_parquet("preprocessed_dataset/sequences_meta_data.parquet")
    train_idx, validation_idx = stratified_group_train_test_split(
        len(full_dataset),
        seq_meta["gesture"],
        seq_meta["subject"],
    )
    expert_train_idx, gating_train_idx = stratified_group_train_test_split(
        len(train_idx),
        seq_meta["gesture"].iloc[train_idx],
        seq_meta["subject"].iloc[train_idx]
    )
    
    return (
        Subset(full_dataset, expert_train_idx),
        Subset(full_dataset, gating_train_idx),
        Subset(full_dataset, validation_idx),
    )