import math
from typing import Optional
from itertools import pairwise, starmap

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from config import *

class MultiScaleConvs(nn.Module):
    def __init__(self, in_channels:int, kernel_sizes:list[int]):
        super().__init__()
        def mk_conv_block(k_size) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(in_channels, in_channels, k_size, padding=k_size // 2, groups=in_channels),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(),
            )
        self.convs = nn.ModuleList(map(mk_conv_block, kernel_sizes))

    def forward(self, x:Tensor) -> Tensor:
        yes = torch.cat([conv(x) for conv in self.convs] + [x], dim=1)
        # print("stem output shape:", yes.shape)
        return yes

class ImuFeatureExtractor(nn.Module):
    def __init__(self, in_channels:int, kernel_size:int=15):
        super().__init__()

        self.lpf = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=in_channels,
            bias=False,
        )
        nn.init.kaiming_uniform_(self.lpf.weight, a=math.sqrt(5))

    def forward(self, x:Tensor) -> Tensor:
        lpf_output = self.lpf(x)
        hpf_output = x - lpf_output
        return torch.cat((lpf_output, hpf_output, x), dim=1)  # (B, C_out, T)

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

class MBConvBlock(nn.Module):
    # From this schema: https://media.licdn.com/dms/image/v2/D5612AQFjbDOm5uyxdw/article-inline_image-shrink_1500_2232/article-inline_image-shrink_1500_2232/0/1683677500817?e=1758153600&v=beta&t=n48_UW5TZTyDPhRFlJXSidUQQPQpuC756M0kNeKmYTY
    def __init__(self, in_chns:int, out_chns:int, se_reduction:int=8, expansion_ratio:int=4, dropout_ratio:float=0.3):
        super().__init__()
        expanded_channels = in_chns * expansion_ratio
        self.blocks = nn.Sequential(
            nn.Conv1d(in_chns, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(expanded_channels),
            nn.ReLU(),
            nn.Conv1d(
                expanded_channels,
                expanded_channels,
                kernel_size=3,
                padding=1,
                groups=expanded_channels,
                bias=False,
            ),
            nn.BatchNorm1d(expanded_channels),
            nn.ReLU(),
            SqueezeExcitationBlock(expanded_channels, se_reduction),
            nn.Conv1d(expanded_channels, out_chns, kernel_size=1, bias=False)
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(out_chns)
            # nn.ReLU(),
            # nn.Dropout(dropout_ratio),
        )
        if in_chns == out_chns:
            self.skip_connection = nn.Identity() 
        else:
            # TODO: set bias to False ?
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_chns, out_chns, 1, bias=False),
                nn.BatchNorm1d(out_chns)
            )
            self.head.add_module("max_pool", nn.MaxPool1d(2))
            
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
            input_meta_data:dict,
            mlp_width:int,
            dataset_x:Optional[Tensor]=None,
            tof_dropout_ratio:float=0,
            thm_dropout_ratio:float=0,
            imu_dropout_ratio:float=0,
        ):
        super().__init__()
        self.input_meta_data = input_meta_data
        if dataset_x is not None:
            x_mean = dataset_x.mean(dim=(0, 2), keepdim=True)
            x_std = dataset_x.std(dim=(0, 2), keepdim=True)
            self.register_buffer("x_mean", x_mean)
            self.register_buffer("x_std", x_std)
        else:
            x_stats_size = (1, len(input_meta_data["feature_cols"]), 1)
            self.register_buffer("x_mean", torch.empty(x_stats_size))
            self.register_buffer("x_std", torch.empty(x_stats_size))
        self.imu_branch = nn.Sequential(
            ResidualBlock(len(input_meta_data["imu_idx"]), 219, imu_dropout_ratio),
            ResidualBlock(219, 500, imu_dropout_ratio),
        )
        self.tof_branch = AlexNet([len(input_meta_data["tof_idx"]), 82, 500], tof_dropout_ratio)
        self.thm_branch = AlexNet([len(input_meta_data["thm_idx"]), 82, 500], thm_dropout_ratio)
        self.rnn = nn.GRU(500 * 3, mlp_width // 2, bidirectional=True)
        self.attention = AdditiveAttentionLayer(mlp_width)
        self.main_head = MLPhead(mlp_width, 18)
        self.aux_orientation_head = MLPhead(mlp_width, input_meta_data["n_aux_classes"])
        self.aux_demographics_head = MLPhead(mlp_width, 2)

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
        return self.main_head(attended), self.aux_orientation_head(attended), self.aux_demographics_head(attended)

def mk_model(
        input_meta_data:dict,
        dataset_x:Optional[Tensor]=None,
        device:Optional[torch.device]=None,
    ) -> nn.Module:
    model = CMIHARModule(
        input_meta_data,
        mlp_width=256,
        dataset_x=dataset_x,
        imu_dropout_ratio=0.2,
        tof_dropout_ratio=0.2,
        thm_dropout_ratio=0.2,
    )
    if device is not None:
        model = model.to(device)
    return model
