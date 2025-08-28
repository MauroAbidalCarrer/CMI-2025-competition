from os.path import join
from functools import partial
from tqdm.notebook import tqdm
from operator import methodcaller
from collections import defaultdict
from typing import Optional, Literal, Iterator
from itertools import pairwise, starmap, product

import torch
import numpy as np
import pandas as pd
from numpy import ndarray
from torch import nn, Tensor
from pandas import DataFrame as DF
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

class CMIDatasetSubset(TensorDataset):
    def __getitem__(self, index):
        return *super().__getitem__(index), index

def stratified_group_train_test_split(
    n_samples: int,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.1,
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

def copy_subset(dataset: TensorDataset, indices) -> TensorDataset:
    cpy_tensor = lambda t: t[indices].to(copy=True)
    return TensorDataset(*tuple(map(cpy_tensor, dataset.tensors)))

def split_dataset() -> dict[str, tuple[TensorDataset, DF]]:
    """Returns: experts training idx, gating model training idx, validation idx"""
    seed_everything(SEED)
    full_dataset = CMIDataset(torch.device("cpu"))
    seq_meta = pd.read_parquet("preprocessed_dataset/sequences_meta_data.parquet")
    train_idx, validation_idx = stratified_group_train_test_split(
        len(full_dataset),
        seq_meta["gesture"],
        seq_meta["subject"],
    )
    train_dataset = copy_subset(full_dataset, train_idx)
    train_seq_meta = seq_meta.iloc[train_idx]
    expert_train_idx, gating_train_idx = stratified_group_train_test_split(
        len(train_dataset),
        train_seq_meta["gesture"],
        train_seq_meta["subject"],
    )

    expert_train = copy_subset(train_dataset, expert_train_idx)
    expert_train = CMIDatasetSubset(*expert_train.tensors)
    return {
        "expert_train": (expert_train, train_seq_meta.iloc[expert_train_idx]),
        "gating_train": (copy_subset(train_dataset, gating_train_idx), train_seq_meta.iloc[gating_train_idx]),
        "validation": (copy_subset(full_dataset, validation_idx), seq_meta.iloc[validation_idx]),
    }

def sgkf_cmi_dataset(dataset: Dataset, seq_meta: DF, n_splits: int) -> Iterator[tuple[int, int, int]]:
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
    )

    fold_indices = list(sgkf.split(np.empty(len(dataset)), seq_meta["gesture"], seq_meta["subject"]))
    folds_idx_oredered_by_score:list[int] = FOLDS_VAL_SCORE_ORDER.get(n_splits, range(n_splits))

    for fold_idx in folds_idx_oredered_by_score:
        yield *fold_indices[fold_idx], SEED + fold_idx

if __name__ == "__main__":
    dataset_splits = split_dataset()
    print(dataset_splits["expert_train"])
    print(dataset_splits["gating_train"])
    print(dataset_splits["validation"])