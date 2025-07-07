import os
from typing import Literal, Optional

import torch
from torch import nn
import pandas as pd
from pandas import DataFrame as DF
from kagglehub import competition_download

from config import COMPETITION_HANDLE


class CMIdataset():
    def __init__(self, split:Literal["train", "test"], nb_sequences_to_load:Optional[int]=None):
        super().__init__()
        if not split in ["train", "test"]:
            raise ValueError(f'"split" argument should be either "train" or "test, got {split}')
        csv_file_path = competition_download(COMPETITION_HANDLE, path=f"{split}.csv")
        self.df = pd.read_csv(csv_file_path)

    def __len__(self):
        return len(self.df)
    