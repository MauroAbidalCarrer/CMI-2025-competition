import math
from typing import Optional
from itertools import pairwise, starmap

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from preprocessing import get_meta_data
from config import *

class SqueezeExcitationBlock(nn.Module):
    # Copy/paste of https://www.kaggle.com/code/wasupandceacar/lb-0-82-5fold-single-bert-model#Model implementation
    def __init__(self, channels:int, reduction:int=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        se = F.adaptive_avg_pool1d(x, 1).squeeze(-1)      # -> (B, C)
        se = F.relu(self.fc1(se), inplace=True)          # -> (B, C//r)
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)    # -> (B, C, 1)
        return x * se

class ResidualBlock(nn.Module):
    def __init__(self, in_chns:int, out_chns:int, dropout_ratio:float=0.3, se_reduction:int=8, kernel_size:int=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(in_chns, out_chns, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_chns),
            nn.ReLU(),
            nn.Conv1d(out_chns, out_chns, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_chns),
            SqueezeExcitationBlock(out_chns, se_reduction),
        )
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout_ratio))
        if in_chns == out_chns:
            self.skip_connection = nn.Identity() 
        else:
            # TODO: set bias to False ?
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_chns, out_chns, 1, bias=False),
                nn.BatchNorm1d(out_chns)
            )
            self.head.insert(1, nn.MaxPool1d(2))

    def forward(self, x:Tensor) -> Tensor:
        activaition_maps = self.skip_connection(x) + self.blocks(x)
        return self.head(activaition_maps)

class AdditiveAttentionLayer(nn.Module):
    # Copied (and slightly modified) from https://www.kaggle.com/code/myso1987/cmi3-pyroch-baseline-model-add-aug-folds
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (batch, channels, seq_len)
        x = x.swapaxes(1, 2)
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context

class AlexNet(nn.Sequential):
    def __init__(self, channels:list[int], dropout_ratio:float):
        def mk_conv_block(in_channels:int, out_channels:int) -> nn.Module:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_ratio),
            )
        return super().__init__(*list(starmap(mk_conv_block, pairwise(channels))))

class MLPhead(nn.Sequential):
    def __init__(self, width:int, n_classes:int):
        super().__init__(
                nn.LazyLinear(width, bias=False),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Linear(width, width // 2, bias=False),
                nn.BatchNorm1d(width // 2),
                nn.ReLU(),
                nn.Linear(width // 2, n_classes),
        )

class CMIHARModule(nn.Module):
    def __init__(
            self,
            mlp_width:int,
            dataset_x:Optional[Tensor]=None,
            reg_demos_dataset_y:Optional[Tensor]=None,
            tof_dropout_ratio:float=0,
            thm_dropout_ratio:float=0,
            imu_dropout_ratio:float=0,
        ):
        super().__init__()
        get_meta_data
        self.input_meta_data = get_meta_data()
        self.init_std_mean(dataset_x, (0, 2), (1, len(self.input_meta_data["feature_cols"]), 1), "x")
        self.init_std_mean(reg_demos_dataset_y, 0, (1, len(REGRES_DEMOS_TARGETS)), "reg_demos_y")
        self.imu_branch = nn.Sequential(
            ResidualBlock(len(self.input_meta_data["imu_idx"]), 219, imu_dropout_ratio),
            ResidualBlock(219, 500, imu_dropout_ratio),
        )
        self.tof_branch = AlexNet([len(self.input_meta_data["tof_idx"]), 82, 500], tof_dropout_ratio)
        self.thm_branch = AlexNet([len(self.input_meta_data["thm_idx"]), 82, 500], thm_dropout_ratio)
        self.rnn = nn.GRU(500 * 3, mlp_width // 2, bidirectional=True)
        self.attention = AdditiveAttentionLayer(mlp_width)
        self.main_head = MLPhead(mlp_width, 18)
        self.aux_orientation_head = MLPhead(mlp_width, self.input_meta_data["n_orient_classes"])
        self.binary_demographics_head = MLPhead(mlp_width, len(BINARY_DEMOS_TARGETS))
        self.regres_demographics_head = MLPhead(mlp_width, len(REGRES_DEMOS_TARGETS))
    
    def init_std_mean(self, data:Optional[Tensor], stats_dim:int|tuple, stats_shape:tuple[int], preffix:str):
        if data is not None:
            mean = data.mean(dim=stats_dim, keepdim=True)
            std = data.std(dim=stats_dim, keepdim=True)
            self.register_buffer(preffix + "_mean", mean)
            self.register_buffer(preffix + "_std", std)
        else:
            self.register_buffer(preffix + "_mean", torch.empty_like(stats_shape))
            self.register_buffer(preffix + "_std", torch.empty_like(stats_shape))
    
    def forward(self, x:Tensor) -> Tensor:
        assert self.x_mean is not None and self.x_std is not None, f"Nor x_mean nor x_std should be None.\nx_std: {self.x_std}\nx_mean: {self.x_mean}"
        x = (x - self.x_mean) / self.x_std
        concatenated_activation_maps = torch.cat(
            (
                self.imu_branch(x[:, self.input_meta_data["imu_idx"]]),
                self.thm_branch(x[:, self.input_meta_data["thm_idx"]]),
                self.tof_branch(x[:, self.input_meta_data["tof_idx"]]),
            ),
            dim=CHANNELS_DIMENSION,
        )
        lstm_output, _  = self.rnn(concatenated_activation_maps.swapaxes(1, 2))
        lstm_output = lstm_output.swapaxes(1, 2) # redundant
        attended = self.attention(lstm_output)
        return (
            self.main_head(attended),
            self.aux_orientation_head(attended),
            self.binary_demographics_head(attended),
            (self.regres_demographics_head(attended) * self.reg_demos_y_std) + self.reg_demos_y_mean,
        )

def mk_model(
        dataset_x:Optional[Tensor]=None,
        reg_demos_dataset_y:Optional[Tensor]=None,
        device:Optional[torch.device]=None,
    ) -> nn.Module:
    model = CMIHARModule(
        mlp_width=256,
        dataset_x=dataset_x,
        reg_demos_dataset_y=reg_demos_dataset_y,
        imu_dropout_ratio=0.2,
        tof_dropout_ratio=0.2,
        thm_dropout_ratio=0.2,
    )
    if device is not None:
        model = model.to(device)
    return model
