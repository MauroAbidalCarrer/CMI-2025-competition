import copy
from os.path import join
from typing import Optional
from functools import partial
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
    def __init__(self, channels:list[int], dropout_ratio:float, groups:int=1):
        def mk_conv_block(in_channels:int, out_channels:int) -> nn.Module:
            return nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 3, padding=1, groups=groups, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.MaxPool1d(2),
                nn.Dropout(dropout_ratio),
            )
        return super().__init__(*list(starmap(mk_conv_block, pairwise(channels))))

class MLPhead(nn.Sequential):
    def __init__(self, width:int, n_classes:int, dropout_ratio:float=0):
        super().__init__(
                nn.LazyLinear(width, bias=False),
                nn.BatchNorm1d(width),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Linear(width, width // 2, bias=False),
                nn.BatchNorm1d(width // 2),
                nn.ReLU(),
                nn.Linear(width // 2, n_classes),
        )

class GaussianNoise(nn.Module):
    """Add Gaussian noise to input tensor"""
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x

class CMIHARModule(nn.Module):
    def __init__(
            self,
            model_kw: dict,
            dataset_x:Optional[Tensor]=None,
            reg_demos_dataset_y:Optional[Tensor]=None,
        ):
        super().__init__()
        get_meta_data
        self.meta_data = get_meta_data().copy()
        self.meta_data["imu_idx"] = np.concatenate((self.meta_data["imu_idx"], self.meta_data["imu_idx"] + self.meta_data["n_features"]))
        self.meta_data["tof_idx"] = np.concatenate((self.meta_data["tof_idx"], self.meta_data["tof_idx"] + self.meta_data["n_features"]))
        self.meta_data["thm_idx"] = np.concatenate((self.meta_data["thm_idx"], self.meta_data["thm_idx"] + self.meta_data["n_features"]))
        if dataset_x is not None:
            self.compute_x_std_and_mean(dataset_x)
        else:
            x_stats_size = (1, len(self.meta_data["feature_cols"]) * 2, 1)
            self.register_buffer("x_mean", torch.empty(x_stats_size))
            self.register_buffer("x_std", torch.empty(x_stats_size))
        self.init_std_mean(reg_demos_dataset_y, 0, (1, len(REGRES_DEMOS_TARGETS)), "reg_demos_y")
        self.imu_branch = nn.Sequential(
            ResidualBlock(len(self.meta_data["imu_idx"]), 219, model_kw["imu_dropout_ratio"]),
            ResidualBlock(219, 500, model_kw["imu_dropout_ratio"]),
        )
        self.tof_branch = AlexNet([len(self.meta_data["tof_idx"]), 100, 500], model_kw["tof_dropout_ratio"], groups=N_TOF_SENSORS)
        self.thm_branch = AlexNet([len(self.meta_data["thm_idx"]), 100, 500], model_kw["thm_dropout_ratio"], groups=N_THM_SENSORS)
        self.rnn = nn.GRU(500 * 3, model_kw["mlp_width"] // 2, bidirectional=True)
        self.rnn_noise = GaussianNoise(model_kw["rnn_gaussian_noise"])
        self.attention = AdditiveAttentionLayer(model_kw["mlp_width"])
        self.bfrb_targets_head = MLPhead(model_kw["mlp_width"], len(BFRB_GESTURES), model_kw["head_dropout_ratio"])
        self.non_bfrb_targets_head = MLPhead(model_kw["mlp_width"], len(NON_BFRB_GESTURES), model_kw["head_dropout_ratio"])
        self.aux_orientation_head = MLPhead(model_kw["mlp_width"], self.meta_data["n_orient_classes"], model_kw["head_dropout_ratio"])
        self.binary_demographics_head = MLPhead(model_kw["mlp_width"], len(BINARY_DEMOS_TARGETS), model_kw["head_dropout_ratio"])
        self.regres_demographics_head = MLPhead(model_kw["mlp_width"], len(REGRES_DEMOS_TARGETS), model_kw["head_dropout_ratio"])

    def compute_x_std_and_mean(self, dataset_x: Tensor):
        x_mean = dataset_x.mean(dim=(0, 2), keepdim=True)
        x_std = dataset_x.std(dim=(0, 2), keepdim=True)
        diff_means = []
        diff_stds = []
        for chan_idx in range(dataset_x.shape[CHANNELS_DIMENSION]):
            diff = dataset_x[:, [chan_idx], 1:] - dataset_x[:, [chan_idx], :-1]
            diff_means.append(diff.mean(dim=(0, 2), keepdim=True))
            diff_stds.append(diff.std(dim=(0, 2), keepdim=True))
        diff_means = torch.concatenate(diff_means, dim=CHANNELS_DIMENSION)
        x_mean = torch.concatenate((x_mean, diff_means), dim=CHANNELS_DIMENSION)
        diff_stds = torch.concatenate(diff_stds, dim=CHANNELS_DIMENSION)
        x_std = torch.concatenate((x_std, diff_stds), dim=CHANNELS_DIMENSION)
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_std", x_std)

    def init_std_mean(self, data:Optional[Tensor], stats_dim:int|tuple, stats_shape:tuple[int], preffix:str):
        if data is not None:
            mean = data.mean(dim=stats_dim, keepdim=True)
            std = data.std(dim=stats_dim, keepdim=True)
            self.register_buffer(preffix + "_mean", mean)
            self.register_buffer(preffix + "_std", std)
        else:
            self.register_buffer(preffix + "_mean", torch.empty(stats_shape))
            self.register_buffer(preffix + "_std", torch.empty(stats_shape))

    def forward(self, x:Tensor) -> Tensor:
        assert self.x_mean is not None and self.x_std is not None, f"Nor x_mean nor x_std should be None.\nx_std: {self.x_std}\nx_mean: {self.x_mean}"
        x = torch.concatenate((
                x, 
                nn.functional.pad(x[..., 1:] - x[..., :-1], (0, 1))
            ),
            dim=CHANNELS_DIMENSION,
        )
        x = (x - self.x_mean) / self.x_std
        concatenated_activation_maps = torch.cat(
            (
                self.imu_branch(x[:, self.meta_data["imu_idx"]]),
                self.thm_branch(x[:, self.meta_data["thm_idx"]]),
                self.tof_branch(x[:, self.meta_data["tof_idx"]]),
            ),
            dim=CHANNELS_DIMENSION,
        )
        lstm_output, _  = self.rnn(concatenated_activation_maps.swapaxes(1, 2))
        lstm_output = lstm_output.swapaxes(1, 2) # redundant
        lstm_output = self.rnn_noise(lstm_output)
        attended = self.attention(lstm_output)
        return (
            torch.concat(
                (
                    self.bfrb_targets_head(attended),
                    self.non_bfrb_targets_head(attended),
                ),
                dim=1
            ),
            self.aux_orientation_head(attended),
            self.binary_demographics_head(attended),
            (self.regres_demographics_head(attended) * self.reg_demos_y_std) + self.reg_demos_y_mean,
        )

def mk_model(
        dataset_x:Optional[Tensor]=None,
        reg_demos_dataset_y:Optional[Tensor]=None,
        device:Optional[torch.device]=None,
        model_kw: Optional[dict]=DFLT_MODEL_HP_KW,
    ) -> nn.Module:
    model = CMIHARModule(
        dataset_x=dataset_x,
        reg_demos_dataset_y=reg_demos_dataset_y,
        model_kw=model_kw,
    )
    if device is not None:
        model = model.to(device)
    return model

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        self.ema_model.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema_model.state_dict().items():
                model_v = msd[k].detach()
                if model_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
                else:
                    ema_v.copy_(model_v)


class ModelEnsemble(nn.ModuleList):
    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        outputs: list[tuple[Tensor, ...]] = [model(x) for model in self]
        outputs: tuple[Tensor] = tuple(map(torch.stack, zip(*outputs)))
        outputs: tuple[Tensor] = tuple(map(partial(torch.mean, dim=0), outputs))
        return outputs
        
def mk_model_ensemble(parent_dir: str, device: torch.device, model_kw=DFLT_MODEL_HP_KW) -> ModelEnsemble:
    models = []
    for fold_idx in range(N_FOLDS):
        model = mk_model(**model_kw).to(device)
        checkpoint = torch.load(
            join(
                parent_dir,
                f"model_fold_{fold_idx}.pth"
            ),
            map_location=device,
            weights_only=True
        )
        model.load_state_dict(checkpoint)
        models.append(model)
    return ModelEnsemble(models).to(device)