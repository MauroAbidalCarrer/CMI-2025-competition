# %% [markdown]
# # Training & inference notebook
# Credit to [Tarun Mishra](https://www.kaggle.com/tarundirector) – this code is heavily based on his [notebook](https://www.kaggle.com/code/tarundirector/sensor-pulse-viz-eda-for-bfrb-detection?scriptVersionId=243465321).

# %% [markdown]
# ## Setup

# %% [markdown]
# ### imports

# %%
import gc
import re
import os
import json 
import math
import shutil
import random
import warnings
from os.path import join
from functools import partial
from tqdm.notebook import tqdm
from collections import defaultdict
from operator import methodcaller
from typing import Optional, Literal
from typing import Optional, Literal, Iterator
from itertools import pairwise, starmap, product

import torch
import optuna
import kagglehub 
import numpy as np
import pandas as pd
import polars as pl
from numpy import ndarray
from torch import nn, Tensor
from numpy.linalg import norm
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.multiprocessing as mp
from pandas import DataFrame as DF
from optuna.trial import TrialState
from sklearn.metrics import f1_score
from optuna.pruners import BasePruner
from optuna.exceptions import TrialPruned
from torch.utils.data import TensorDataset
from scipy.spatial.transform import Rotation
import kaggle_evaluation.cmi_inference_server
from torch.utils.data import DataLoader as DL
from sklearn.model_selection import GroupKFold
from rich.progress import Progress, Task, track
from sklearn.model_selection import train_test_split
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import ConstantLR, LRScheduler, _LRScheduler

# %% [markdown]
# ### Configs

# %%
# Dataset
COMPETITION_HANDLE = "cmi-detect-behavior-with-sensor-data"
TARGET_NAMES = sorted([
    "Above ear - pull hair",
    "Cheek - pinch skin",
    "Eyebrow - pull hair",
    "Eyelash - pull hair",
    "Feel around in tray and pull out an object",
    "Forehead - pull hairline",
    "Forehead - scratch",
    "Neck - pinch skin",
    "Neck - scratch",
    "Text on phone",
    "Wave hello",
    "Write name in air",
    "Write name on leg",
    "Drink from bottle/cup",
    "Pinch knee/leg skin",
    "Pull air toward your face",
    "Scratch knee/leg skin",
    "Glasses on/off"
])
BFRB_GESTURES = [
    'Above ear - pull hair',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Neck - pinch skin',
    'Neck - scratch',
    'Cheek - pinch skin'
]
BFRB_INDICES = [idx for idx, gesture in enumerate(TARGET_NAMES) if gesture in BFRB_GESTURES]
IMU_FEATS_PREFIXES = (
    "acc",
    "linear_acc",
    "rot",
    "angular",
    "euler",
    "quat_rot_mag",
    "delta_rot_mag",
)
QUATERNION_COLS = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
GRAVITY_WORLD = np.array([0, 0, 9.81], "float32")
RAW_ACCELRATION_COLS = ["acc_x", "acc_y", "acc_z"]
LINEAR_ACC_COLS = ["linear_" + col for col in RAW_ACCELRATION_COLS] # Acceleration without gravity
COMPETITION_HANDLE = "cmi-detect-behavior-with-sensor-data"
CATEGORY_COLUMNS = [
    'row_id',
    'sequence_type',
    'sequence_id',
    'subject',
    'orientation',
    'behavior',
    'phase',
    'gesture',
]
META_DATA_COLUMNS = [
    'row_id',
    'sequence_type',
    'sequence_id',
    'sequence_counter',
    'subject',
    'orientation',
    'behavior',
    'phase',
    'gesture',
]
DATASET_DF_DTYPES = {
    "acc_x": "float32", "acc_y": "float32", "acc_z": "float32",
    "thm_1":"float32", "thm_2":"float32", "thm_3":"float32", "thm_4":"float32", "thm_5":"float32",
    "sequence_counter": "int32",
    **{col: "category" for col in CATEGORY_COLUMNS},
    **{f"tof_{i_1}_v{i_2}": "float32" for i_1, i_2 in product(range(1, 5), range(64))},
}
PREPROCESSED_DATASET_HANDLE = "mauroabidalcarrer/prepocessed-cmi-2025"
# The quantile of the sequences len used to pad/truncate during preprocessing
SEQUENCE_NORMED_LEN_QUANTILE = 0.95
# SAMPLING_FREQUENCY = 10 #Hz
VALIDATION_FRACTION = 0.2
EPSILON=1e-8
DELTA_ROTATION_ANGULAR_VELOCITY_COLS = ["angular_vel_x", "angular_vel_y", "angular_vel_z"]
DELTA_ROTATION_AXES_COLS = ["rotation_axis_x", "rotation_axis_y", "rotation_axis_z"]
EULER_ANGLES_COLS = ["euler_x", "euler_y", "euler_z"]
pad_trunc_mode_type = Literal["pre", "center", "post"]
SEQ_PAD_TRUNC_MODE: pad_trunc_mode_type = "center"
DEFAULT_VERSION_NOTES = "Preprocessed Child Mind Institue 2025 competition preprocessed dataset."
NB_COLS_PER_TOF_SENSOR = 64
TOF_PATCH_SIZE = 2
assert ((NB_COLS_PER_TOF_SENSOR // 2) % TOF_PATCH_SIZE) == 0, "tof side len should be dividable by TOF_PATCH_SIZE!"
TOF_AGG_FUNCTIONS = [
    "mean",
    "std",
    "median",
    "min",
    "max",
]
DEMOS_TARGETS = ["sex", "handedness"]
# Data augmentation
JITTER = 0.25
SCALING = 0.2
MIXUP = 0.3
LABEL_SMOOTHING = 0.1
# Training loop
N_FOLDS = 10
TRAIN_BATCH_SIZE = 256
VALIDATION_BATCH_SIZE = 4 * TRAIN_BATCH_SIZE
PATIENCE = 8
# Optimizer
WEIGHT_DECAY = 3e-3
# Scheduler
TRAINING_EPOCHS = 35 # Including warmup epochs
WARMUP_EPOCHS = 3
WARMUP_LR_INIT = 1.822126131809773e-05
MAX_TO_MIN_LR_DIV_FACTOR = 100
LR_CYCLE_FACTOR = 0.5
CYCLE_LENGTH_FACTOR = 0.9
INIT_CYCLE_EPOCHS = 6
CHANNELS_DIMENSION = 1
SEED = 42
FOLDS_VAL_SCORE_ORDER = {
    10: [4, 7, 1, 9, 6, 2, 3, 8, 0, 5],
    5: [3, 1, 4, 2, 0],
}
# model
KAGGLE_USERNAME = "mauroabidalcarrer"
MODEL_NAME = "cmi-model"
MODEL_VARIATION = "single_model_architecture"

# %% [markdown]
# ### Seed everything

# %%
def seed_everything(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)
seed_everything(seed=SEED)

# %% [markdown]
# ### Supress performance warngings

# %%
warnings.filterwarnings(
    "ignore",
    message=(
        "DataFrame is highly fragmented.  This is usually the result of "
        "calling `frame.insert` many times.*"
    ),
    category=pd.errors.PerformanceWarning,
)
warnings.filterwarnings("ignore", message=".*sm_120.*")

# %% [markdown]
# ### device setup

# %%
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device

# %% [markdown]
# ## Dataset

# %% [markdown]
# ### Preprocessing

# %%
def get_feature_cols(df:DF) -> list[str]:
    return sorted(list(set(df.columns) - set(META_DATA_COLUMNS) - set(TARGET_NAMES)))

# Missing ToF values are already imputed by -1 which is inconvinient since we want all missing values to be NaN.    
# So we replace them by NaN and then perform imputing.
def get_fillna_val_per_feature_col(df:DF) -> dict:
    return {col: 1.0 if col == 'rot_w' else 0 for col in get_feature_cols(df)}

def imputed_features(df:DF) -> DF:
    # Missing ToF values are already imputed by -1 which is inconvinient since we want all missing values to be NaN.    
    # So we replace them by NaN and then perform imputing.  
    tof_vals_to_nan = {col: -1.0 for col in df.columns if col.startswith("tof")}
    # fillna_val_per_col = {col: 1.0 if col == 'rot_w' else 0 for col in df.columns}

    df[get_feature_cols(df)] = (
        df
        .loc[:, get_feature_cols(df)]
        # df.replace with np.nan sets dtype to floar64 so we set it back to float32
        .replace(tof_vals_to_nan, value=np.nan)
        .astype("float32")
        .groupby(df["sequence_id"], observed=True, as_index=False)
        .ffill()
        .groupby(df["sequence_id"], observed=True, as_index=False)
        .bfill()
        # In case there are only nan in the column in the sequence
        .fillna(get_fillna_val_per_feature_col(df))
    )
    return df

def standardize_tof_cols_names(df: DF) -> DF:
    renamed_cols = {}
    pattern = re.compile(r"^(tof_\d_v)(\d)$")  # match 'tof_X_vY' where Y is a single digit

    for col in df.columns:
        match = pattern.match(col)
        if match:
            prefix, version = match.groups()
            new_col = f"{prefix}0{version}"
            renamed_cols[col] = new_col

    return df.rename(columns=renamed_cols)

def norm_quat_rotations(df:DF) -> DF:
    df[QUATERNION_COLS] /= np.linalg.norm(df[QUATERNION_COLS], axis=1, keepdims=True)
    return df

def add_linear_acc_cols(df:DF) -> DF:
    # Vectorized version of https://www.kaggle.com/code/wasupandceacar/lb-0-82-5fold-single-bert-model#Dataset `remove_gravity_from_acc`
    rotations:Rotation = Rotation.from_quat(df[QUATERNION_COLS])
    gravity_sensor_frame = rotations.apply(GRAVITY_WORLD, inverse=True).astype("float32")
    df[LINEAR_ACC_COLS] = df[RAW_ACCELRATION_COLS] - gravity_sensor_frame
    return df

def add_acc_magnitude(df:DF, acc_cols:list[str], acc_mag_col_name:str) -> DF:
    return df.assign(**{acc_mag_col_name: np.linalg.norm(df.loc[:, acc_cols], axis=1)})

def add_quat_angle_mag(df:DF) -> DF:
    return df.assign(quat_rot_mag=np.arccos(df["rot_w"]) * 2)

def add_angular_velocity_features(df:DF) -> DF:
    rotations = Rotation.from_quat(df[QUATERNION_COLS])
    delta_rotations = rotations[1:] * rotations[:-1].inv()
    delta_rot_velocity = delta_rotations.as_rotvec()
    # Add extra line to avoid shape mismatch
    delta_rot_velocity = np.vstack((np.zeros((1, 3)), delta_rot_velocity))
    delta_rot_magnitude = norm(delta_rot_velocity, axis=1, keepdims=True)
    delta_rot_axes = delta_rot_velocity / (delta_rot_magnitude + EPSILON)
    df[DELTA_ROTATION_ANGULAR_VELOCITY_COLS] = delta_rot_velocity
    df[DELTA_ROTATION_AXES_COLS] = delta_rot_axes
    df["delta_rot_mag"] = delta_rot_magnitude.squeeze()

    return df

def rot_euler_angles(df:DF) -> ndarray:
    df[EULER_ANGLES_COLS] = (
        Rotation
        .from_quat(df[QUATERNION_COLS])
        .as_euler("xyz")
        .squeeze()
    )
    return df

def agg_tof_patch(tof_views:np.ndarray, f_name:str) -> ndarray:
    views_agg_func = methodcaller(f_name, tof_views, axis=(1, 2))
    return (
        views_agg_func(np)
        .reshape(tof_views.shape[0], -1)
    )

def agg_tof_cols_per_sensor(df:DF) -> DF:
    """
    ## Description:
    Computes the sensor and patch sensor wise stats.
    ## Resturns:
    The dataframe with the added stats.
    """
    for tof_idx in tqdm(range(1, 6)):
        tof_name = f"tof_{tof_idx}"
        all_tof_cols = [f"{tof_name}_v{v_idx:02d}" for v_idx in range(64)]
        tof_feats = (
            df
            .loc[:, all_tof_cols]
            .values
            .reshape(-1, 8, 8)
        )
        agg_func = partial(df[all_tof_cols].agg, axis="columns")
        mk_fe_col_name = lambda f_name: tof_name + "_" + f_name
        engineered_feats = DF({mk_fe_col_name(f_name): agg_func(f_name) for f_name in TOF_AGG_FUNCTIONS})
        stats_cols_names = list(map(mk_fe_col_name, TOF_AGG_FUNCTIONS))
        # Patch Feature engineering
        tof_views:np.ndarray = sliding_window_view(tof_feats, (TOF_PATCH_SIZE, TOF_PATCH_SIZE), (1, 2))
        patch_fe = {}
        for f_name in TOF_AGG_FUNCTIONS:
            tof_patch_stats = agg_tof_patch(tof_views, f_name)
            for patch_idx in range(tof_patch_stats.shape[1]):
                key = mk_fe_col_name(f_name) + f"_{patch_idx:02d}"
                patch_fe[key] = tof_patch_stats[:, patch_idx]
        patch_df = DF(patch_fe)
        # concat results
        df = pd.concat(
            (
                df.drop(columns=filter(df.columns.__contains__, stats_cols_names)),
                engineered_feats,
                patch_df,
            ),
            axis="columns",
        )
    return df

def add_diff_features(df:DF) -> DF:
    return pd.concat(
        (
            df,
            (
                df
                .groupby("sequence_id", as_index=False, observed=True)
                [get_feature_cols(df)]
                .diff()
                .fillna(get_fillna_val_per_feature_col(df))
                .add_suffix("_diff")
            )
        ),
        axis="columns",
    )

def one_hot_encode_targets(df:DF) -> DF:
    one_hot_target = pd.get_dummies(df["gesture"], dtype="float32")
    df[TARGET_NAMES] = one_hot_target[TARGET_NAMES]
    return df

def length_normed_sequence_feat_arr(
        sequence: DF,
        normed_sequence_len: int,
        SEQ_PAD_TRUNC_MODE:Literal["pre", "center", "post"]
    ) -> ndarray:
    features = (
        sequence
        .loc[:, get_feature_cols(sequence)]
        .values
    )
    len_diff = abs(normed_sequence_len - len(features))
    len_diff_h = len_diff // 2 # half len diff
    len_diff_r = len_diff % 2 # len diff remainder
    if len(features) < normed_sequence_len:
        padding_dict = {
            "pre": (len_diff, 0),
            "center": (len_diff_h + len_diff_r, len_diff_h),
            "post": (0, len_diff),
        }
        padded_features = np.pad(
            features,
            (padding_dict[SEQ_PAD_TRUNC_MODE], (0, 0)),
        )
        return padded_features
    elif len(features) > normed_sequence_len:
        truncating_dict = {
            "pre": slice(len_diff),
            "center": slice(len_diff_h, -len_diff_h),
            "post": slice(0, -len_diff),
        }
        return features[len_diff // 2:-len_diff // 2]
    else:
        return features

def df_to_ndarrays(df:DF, normed_sequence_len:int, seq_pad_trunc_mode:str) -> tuple[np.ndarray, np.ndarray]:
    sequence_it = df.groupby("sequence_id", observed=True, as_index=False)
    x = np.empty(
        shape=(len(sequence_it), normed_sequence_len, len(get_feature_cols(df))),
        dtype="float32"
    )
    y = np.empty(
        shape=(len(sequence_it), len(TARGET_NAMES)),
        dtype="float32"
    )
    for sequence_idx, (_, sequence) in tqdm(enumerate(sequence_it), total=len(sequence_it)):
        normed_seq_feat_arr = length_normed_sequence_feat_arr(sequence, normed_sequence_len, seq_pad_trunc_mode)
        x[sequence_idx] = normed_seq_feat_arr
        # Take the first value as they are(or at least should be) all the same in a single sequence
        y[sequence_idx] = sequence[TARGET_NAMES].iloc[0].values

    return x, y

def get_normed_seq_len(dataset:DF) -> int:
    return int(
        dataset
        .groupby("sequence_id", observed=True)
        .size()
        .quantile(SEQUENCE_NORMED_LEN_QUANTILE)
    )

def fold_dfs_to_ndarrays(train:DF, validation:DF, dataset_normed_seq_len:int, seq_pad_trunc_mode:str) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Returns:
        (train X, train Y, validation X, validation Y)
    """
    # full_dataset_normed_seq_len = get_normed_seq_len(df)
    return (
        *df_to_ndarrays(train, dataset_normed_seq_len, seq_pad_trunc_mode),
        *df_to_ndarrays(validation, dataset_normed_seq_len, seq_pad_trunc_mode),
    )

# %%
def preprocess_competitino_dataset() -> DF:
    csv_path = kagglehub.competition_download(COMPETITION_HANDLE, path="train.csv")
    return (
        pd.read_csv(csv_path, dtype=DATASET_DF_DTYPES)
        .pipe(imputed_features)
        .pipe(standardize_tof_cols_names)
        .pipe(norm_quat_rotations)
        .pipe(add_linear_acc_cols)
        .pipe(add_acc_magnitude, RAW_ACCELRATION_COLS, "acc_mag")
        .pipe(add_acc_magnitude, LINEAR_ACC_COLS, "linear_acc_mag")
        .pipe(add_quat_angle_mag)
        .pipe(add_angular_velocity_features)
        .pipe(rot_euler_angles)
        .pipe(add_quat_angle_mag)
        .pipe(one_hot_encode_targets)
        .pipe(agg_tof_cols_per_sensor)
        .pipe(add_diff_features)
    )

def save_sequence_meta_data(df:DF) -> DF:
    demographics_csv_path = kagglehub.competition_download(COMPETITION_HANDLE, path="train_demographics.csv")
    demographics = pd.read_csv(demographics_csv_path)
    seq_grp = df.groupby("sequence_id", as_index=False, observed=True)
    seq_behavior_proportions = (
        seq_grp
        ["behavior"]
        .value_counts(normalize=True)
        .pivot_table(
            index="sequence_id",
            columns="behavior",
            values="proportion",
            observed=True,
        )
        .reset_index()
    )
    seq_meta_data = (
        seq_grp
        [META_DATA_COLUMNS]
        .last()
        .merge(demographics, how="left", on="subject")
        .merge(seq_behavior_proportions, how="left", on="sequence_id")
    )
    seq_meta_data.to_parquet("preprocessed_dataset/sequences_meta_data.parquet")
    np.save(
        "preprocessed_dataset/orientation_Y.npy",
        pd.get_dummies(seq_meta_data["orientation"], dtype="float32").values,
    )
    np.save(
        "preprocessed_dataset/demographics_Y.npy",
        seq_meta_data[["sex", "handedness"]].values.astype("float32"),
    )

def save_df_meta_data(df:DF):
    full_dataset_meta_data = {
        "mean": df[get_feature_cols(df)].mean().astype("float32").to_dict(),
        "std": df[get_feature_cols(df)].std().astype("float32").to_dict(),
        "pad_seq_len": get_normed_seq_len(df),
        "feature_cols": get_feature_cols(df),
        "n_aux_classes": df["orientation"].nunique(),
    }
    with open("preprocessed_dataset/full_dataset_meta_data.json", "w") as fp:
        json.dump(full_dataset_meta_data, fp, indent=4)

def create_preprocessed_dataset():
    shutil.rmtree("preprocessed_dataset", ignore_errors=True)
    os.makedirs("preprocessed_dataset")
    df = preprocess_competitino_dataset()
    df.to_parquet("preprocessed_dataset/df.parquet")
    full_dataset_sequence_length_norm = get_normed_seq_len(df)
    full_x, full_y = df_to_ndarrays(df, full_dataset_sequence_length_norm, SEQ_PAD_TRUNC_MODE)
    np.save(join("preprocessed_dataset", "X.npy"), full_x, allow_pickle=False)
    np.save(join("preprocessed_dataset", "Y.npy"), full_y, allow_pickle=False)
    # Save meta data
    save_sequence_meta_data(df)
    save_df_meta_data(df)

# %% [markdown]
# ### Compute and save preprocessed dataset

# %%
# create_preprocessed_dataset()

# %% [markdown]
# ### Meta data loading

# %%
meta_data_path = join(
    "preprocessed_dataset",
    "full_dataset_meta_data.json"
)
with open(meta_data_path, "r") as fp:
    meta_data = json.load(fp)
seq_meta_data = pd.read_parquet("preprocessed_dataset/sequences_meta_data.parquet")
demographics = pd.read_csv(kagglehub.competition_download(COMPETITION_HANDLE, path="train_demographics.csv"))
# Convert target names into a ndarray to index it batchwise.
def get_sensor_indices(sensor_prefix: str) -> list[int]:
    is_sensor_feat = methodcaller("startswith", sensor_prefix)
    return [feat_idx for feat_idx, feat in enumerate(meta_data["feature_cols"]) if is_sensor_feat(feat)]
tof_idx = get_sensor_indices("tof")
thm_idx = get_sensor_indices("thm")
imu_idx = list(filter(lambda idx: idx not in tof_idx + thm_idx, range(len(meta_data["feature_cols"]))))

# %% [markdown]
# ### Dataset class

# %%
class CMIDataset(TensorDataset):
    def __init__(self, device:torch.device):
        x = np.load(join("preprocessed_dataset", "X.npy")).swapaxes(1, 2)
        y = np.load(join("preprocessed_dataset", "Y.npy"))
        auxiliary_orientation_y = np.load(join("preprocessed_dataset", "orientation_Y.npy"))
        auxiliary_demographics_y = np.load(join("preprocessed_dataset", "demographics_Y.npy"))
        super().__init__(
            torch.from_numpy(x).to(device),
            torch.from_numpy(y).to(device),
            torch.from_numpy(auxiliary_orientation_y).to(device),
            torch.from_numpy(auxiliary_demographics_y).to(device),
        )

    def __getitem__(self, index):
        return *super().__getitem__(index), index

# %% [markdown]
# ## Model definition

# %%
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
            imu_idx:list[int],
            thm_idx:list[int],
            tof_idx:list[int],
            mlp_width:int,
            dataset_x:Optional[Tensor]=None,
            tof_dropout_ratio:float=0,
            thm_dropout_ratio:float=0,
            imu_dropout_ratio:float=0,
        ):
        super().__init__()
        self.imu_idx = imu_idx
        self.tof_idx = tof_idx
        self.thm_idx = thm_idx
        if dataset_x is not None:
            x_mean = dataset_x.mean(dim=(0, 2), keepdim=True)
            x_std = dataset_x.std(dim=(0, 2), keepdim=True)
            self.register_buffer("x_mean", x_mean)
            self.register_buffer("x_std", x_std)
        else:
            x_stats_size = (1, len(meta_data["feature_cols"]), 1)
            self.register_buffer("x_mean", torch.empty(x_stats_size))
            self.register_buffer("x_std", torch.empty(x_stats_size))
        self.imu_branch = nn.Sequential(
            ResidualBlock(len(imu_idx), 219, imu_dropout_ratio),
            ResidualBlock(219, 500, imu_dropout_ratio),
        )
        self.tof_branch = AlexNet([len(tof_idx), 82, 500], tof_dropout_ratio)
        self.thm_branch = AlexNet([len(thm_idx), 82, 500], thm_dropout_ratio)
        self.rnn = nn.GRU(500 * 3, mlp_width // 2, bidirectional=True)
        self.attention = AdditiveAttentionLayer(mlp_width)
        self.main_head = MLPhead(mlp_width, 18)
        self.aux_orientation_head = MLPhead(mlp_width, meta_data["n_aux_classes"])
        self.aux_demographics_head = MLPhead(mlp_width, 2)

    def forward(self, x:Tensor) -> Tensor:
        assert self.x_mean is not None and self.x_std is not None, f"Nor x_mean nor x_std should be None.\nx_std: {self.x_std}\nx_mean: {self.x_mean}"
        x = (x - self.x_mean) / self.x_std
        concatenated_activation_maps = torch.cat(
            (
                self.imu_branch(x[:, self.imu_idx]),
                self.thm_branch(x[:, self.thm_idx]),
                self.tof_branch(x[:, self.tof_idx]),
            ),
            dim=CHANNELS_DIMENSION,
        )
        lstm_output, _  = self.rnn(concatenated_activation_maps.swapaxes(1, 2))
        lstm_output = lstm_output.swapaxes(1, 2) # redundant
        attended = self.attention(lstm_output)
        return self.main_head(attended), self.aux_orientation_head(attended), self.aux_demographics_head(attended)

# %% [markdown]
# ### Create model function

# %%
def mk_model(
        dataset_x:Optional[Tensor]=None,
        device:Optional[torch.device]=None,
    ) -> nn.Module:
    model = CMIHARModule(
        imu_idx=imu_idx,
        thm_idx=thm_idx,
        tof_idx=tof_idx,
        mlp_width=256,
        dataset_x=dataset_x,
        imu_dropout_ratio=0.2,
        tof_dropout_ratio=0.2,
        thm_dropout_ratio=0.2,
    )
    if device is not None:
        model = model.to(device)
    return model


# %% [markdown]
# ## Training

# %%
class CosineAnnealingWarmupRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_lr: float,
        min_lr: float,
        cycle_length: int,
        cycle_mult: float = 1.0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer: Wrapped optimizer.
            warmup_steps: Number of steps for linear warmup.
            max_lr: Initial maximum learning rate.
            min_lr: Minimum learning rate after decay.
            cycle_length: Initial number of steps per cosine cycle.
            cycle_mult: Multiplicative factor for increasing cycle lengths.
            gamma: Multiplicative decay factor for max_lr after each cycle.
            last_epoch: The index of last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_length = cycle_length
        self.cycle_mult = cycle_mult
        self.gamma = gamma

        self.current_cycle = 0
        self.cycle_step = 0
        self.lr = max_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = (self.last_epoch + 1) / self.warmup_steps
            return [self.min_lr + scale * (self.max_lr - self.min_lr) for _ in self.base_lrs]

        # Adjust for post-warmup step index
        t = self.cycle_step
        T = self.cycle_length

        cosine_decay = 0.5 * (1 + math.cos(math.pi * t / T))
        lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay

        return [lr for _ in self.base_lrs]

    def step(self, epoch: Optional[int] = None) -> None:
        if self.last_epoch >= self.warmup_steps:
            self.cycle_step += 1
            if self.cycle_step >= self.cycle_length:
                self.current_cycle += 1
                self.cycle_step = 0
                self.cycle_length = max(int(self.cycle_length * self.cycle_mult), 1)
                self.max_lr *= self.gamma
        super().step(epoch)

def mk_scheduler(optimizer:Optimizer, steps_per_epoch:int, lr_scheduler_kw:dict) -> _LRScheduler:
    return CosineAnnealingWarmupRestarts(
        optimizer,
        warmup_steps=lr_scheduler_kw["warmup_epochs"] * steps_per_epoch,
        cycle_mult=lr_scheduler_kw["cycle_mult"],
        max_lr=lr_scheduler_kw["max_lr"],
        min_lr=lr_scheduler_kw["max_lr"] / lr_scheduler_kw["max_to_min_div_factor"],
        cycle_length=lr_scheduler_kw["init_cycle_epochs"] * steps_per_epoch,
        gamma=lr_scheduler_kw["lr_cycle_factor"],
    ) 


# %%
def mixup_data(
    x:Tensor,
    y:Tensor,
    aux_y:Tensor,
    alpha=0.2
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    mixed_aux_y = lam * aux_y + (1 - lam) * aux_y[index, :]
    
    return mixed_x, mixed_y, mixed_aux_y

# %%
def train_model_on_single_epoch(
        model:nn.Module,
        train_loader:DL,
        criterion:callable,
        optimizer:torch.optim.Optimizer,
        scheduler:_LRScheduler,
        training_kw:dict,
        device:torch.device,
    ) -> dict:
    "Train model on a single epoch"
    train_metrics = {}
    model.train()
    train_metrics["train_loss"] = 0.0
    total = 0
    for batch_x, batch_y, batch_orientation_y, demographics_y, idx in train_loader:
        batch_orientation_y = batch_orientation_y.clone()
        batch_x = batch_x.to(device).clone()
        add_noise = torch.randn_like(batch_x, device=device) * 0.04
        scale_noise = torch.rand_like(batch_x, device=device) * (1.1 - 0.9) + 0.9
        batch_x = (add_noise + batch_x) * scale_noise
        batch_x[:TRAIN_BATCH_SIZE // 2, tof_idx + thm_idx] = 0.0
        batch_y = batch_y.to(device)
        batch_x = batch_x.float()
        
        batch_x, batch_y, batch_orientation_y = mixup_data(batch_x, batch_y, batch_orientation_y)

        optimizer.zero_grad()
        outputs, orient_output, demos_output = model(batch_x)
        bce = nn.BCEWithLogitsLoss()
        loss = criterion(outputs, batch_y)  \
               + bce(demos_output, demographics_y) * training_kw["demos_loss_weight"] \
               + criterion(orient_output, batch_orientation_y) * training_kw["orient_loss_weight"] \
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_metrics["train_loss"] += loss.item() * batch_x.size(0)
        total += batch_x.size(0)

    train_metrics["train_loss"] /= total

    return train_metrics

# %%
def evaluate_model(model:nn.Module, validation_loader:DL, criterion:callable, device:torch.device) -> dict:
    model.eval()
    eval_metrics = {}
    eval_metrics["val_loss"] = 0.0
    total = 0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch_x, batch_y, oirent_y, demos_y, idx in validation_loader:
            batch_x = batch_x.to(device).clone()
            batch_y = batch_y.to(device)
            batch_x[:VALIDATION_BATCH_SIZE // 2, tof_idx + thm_idx] = 0.0

            outputs, _, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            eval_metrics["val_loss"] += loss.item() * batch_x.size(0)
            total += batch_x.size(0)

            # Get predicted class indices
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            # Get true class indices from one-hot
            trues = torch.argmax(batch_y, dim=1).cpu().numpy()

            all_true.append(trues)
            all_pred.append(preds)

    eval_metrics["val_loss"] /= total
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)

    # Compute competition metrics
    # Binary classification: BFRB (1) vs non-BFRB (0)
    binary_true = np.isin(all_true, BFRB_INDICES).astype(int)
    binary_pred = np.isin(all_pred, BFRB_INDICES).astype(int)
    eval_metrics["binary_f1"] = f1_score(binary_true, binary_pred)

    # Collapse non-BFRB gestures into a single class
    collapsed_true = np.where(
        np.isin(all_true, BFRB_INDICES),
        all_true,
        len(BFRB_GESTURES)  # Single non-BFRB class
    )
    collapsed_pred = np.where(
        np.isin(all_pred, BFRB_INDICES),
        all_pred,
        len(BFRB_GESTURES)  # Single non-BFRB class
    )

    # Macro F1 on collapsed classes
    eval_metrics["macro_f1"] = f1_score(collapsed_true, collapsed_pred, average='macro')
    eval_metrics["final_metric"] = (eval_metrics["binary_f1"] + eval_metrics["macro_f1"]) / 2

    return eval_metrics

# %%
def get_perf_and_seq_id(model:nn.Module, data_loader:DL, device:torch.device) -> DF:
    metrics:dict[list[ndarray]] = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_aux_y, batch_orient_y, idx in data_loader:
            batch_x = batch_x.to(device).clone()
            batch_y = batch_y.to(device)
            # batch_x[:VALIDATION_BATCH_SIZE // 2, tof_idx + thm_idx] = 0.0

            outputs, orient_outputs, demos_output = model(batch_x)
            losses = nn.functional.cross_entropy(
                outputs,
                batch_y,
                label_smoothing=LABEL_SMOOTHING,
                reduction="none",
            )
            orient_losses = nn.functional.cross_entropy(
                orient_outputs,
                batch_aux_y,
                label_smoothing=LABEL_SMOOTHING,
                reduction="none",
            )
            demos_losses = nn.functional.binary_cross_entropy_with_logits(
                demos_output,
                batch_orient_y,
                # label_smoothing=LABEL_SMOOTHING,
                reduction="none",
            ).cpu().numpy()
            # Get predicted class indices
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            # Get true class indices from one-hot
            trues = torch.argmax(batch_y, dim=1).cpu().numpy()
            accuracies = (preds == trues) #.cpu().numpy()

            metrics["losses"].append(losses.cpu().numpy())
            metrics["orient_losses"].append(orient_losses.cpu().numpy())
            for denos_target_idx, demos_target in enumerate(DEMOS_TARGETS):
                metrics[f"{demos_target}_losses"].append(demos_losses[:, denos_target_idx])
            metrics["preds"].append(preds)
            metrics["trues"].append(trues)
            metrics["accuracies"].append(accuracies)
            metrics["sequence_id"].append(seq_meta_data["sequence_id"].iloc[idx].values)

    metrics = {k: np.concat(v) for k, v in metrics.items()}

    return DF.from_records(metrics)

def get_per_sequence_meta_data(model:nn.Module, train_dataset:Dataset, val_dataset:Dataset, device:torch.device) -> DF:
    train_DL = DL(train_dataset, VALIDATION_BATCH_SIZE, shuffle=False)
    val_DL = DL(val_dataset, VALIDATION_BATCH_SIZE, shuffle=False)
    return (
        pd.concat((
            get_perf_and_seq_id(model, train_DL, device).assign(is_train=True),
            get_perf_and_seq_id(model, val_DL, device).assign(is_train=False),
        ))
        .merge(seq_meta_data, how="left", on="sequence_id")
    )

# %%
def train_model_on_all_epochs(
        model:nn.Module,
        train_dataset:Dataset,
        validation_dataset:Dataset,
        criterion:callable,
        optimizer:torch.optim.Optimizer,
        scheduler:_LRScheduler,
        fold:int,
        training_kw:dict,
        device:torch.device,
    ) -> tuple[DF, DF]:
    """
    Returns:
        tuple[DF, DF]: epoch wise metrics, sample(sequence) wise meta data metrics
    """
    train_loader = DL(train_dataset, TRAIN_BATCH_SIZE, shuffle=True, drop_last=False)
    validation_loader = DL(validation_dataset, VALIDATION_BATCH_SIZE, shuffle=False, drop_last=False)

    metrics:list[dict] = []
    # Early stopping
    best_metric = -np.inf
    epochs_no_improve = 0

    for epoch in range(1, TRAINING_EPOCHS + 1):
        train_metrics = train_model_on_single_epoch(model, train_loader, criterion, optimizer, scheduler, training_kw, device)
        validation_metrics = evaluate_model(model, validation_loader, criterion, device)
        metrics.append({"fold": fold, "epoch": epoch} | train_metrics | validation_metrics)
        print(f"epoch {epoch:02d}: {validation_metrics['final_metric']:.4f}")
        if validation_metrics["final_metric"] > best_metric:
            best_metric = validation_metrics["final_metric"]
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                model.load_state_dict(best_model_state)
                break
    os.makedirs("models", exist_ok=True)
    torch.save(best_model_state, f"models/model_fold_{fold}.pth")
    epoch_wise_metrics = DF.from_records(metrics).set_index(["fold", "epoch"])
    sample_wise_meta_data = get_per_sequence_meta_data(model, train_dataset, validation_dataset, device)

    return epoch_wise_metrics, sample_wise_meta_data

# %%
def train_on_single_fold(
        fold_idx:int,
        full_dataset:TensorDataset,
        train_idx:ndarray,
        validation_idx:ndarray,
        lr_scheduler_kw: dict,
        optimizer_kw: dict,
        training_kw: dict,
        seed:int,
        gpu_id:int,
    ) -> tuple[DF, DF]:
    seed_everything(seed)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    train_dataset = Subset(full_dataset, train_idx)
    validation_dataset = Subset(full_dataset, validation_idx)
    all_train_x = train_dataset.dataset.tensors[0][train_dataset.indices]
    device = torch.device(f"cuda:{gpu_id}")
    model = mk_model(all_train_x, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        WARMUP_LR_INIT,
        weight_decay=optimizer_kw["weight_decay"],
        betas=(optimizer_kw["beta_0"], optimizer_kw["beta_1"]),
    )
    steps_per_epoch = len(DL(train_dataset, TRAIN_BATCH_SIZE)) # ugly, i know
    scheduler = mk_scheduler(optimizer, steps_per_epoch, lr_scheduler_kw)
    return train_model_on_all_epochs(
        model,
        train_dataset,
        validation_dataset,
        criterion,
        optimizer,
        scheduler,
        fold_idx,
        training_kw,
        device=device,
    )

# %%
def sgkf_from_tensor_dataset(
    # dataset: TensorDataset,
    n_splits: int = 5,
    shuffle: bool = True,
) -> Iterator[tuple[Subset, Subset, int]]:
    # Load sequence meta data to get classes and groups parameters
    seq_meta = pd.read_parquet("preprocessed_dataset/sequences_meta_data.parquet")
    # X, *_ = dataset.tensors
    x = np.load("preprocessed_dataset/X.npy")
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
    )

    fold_indices = list(sgkf.split(x, seq_meta["gesture"], seq_meta["subject"]))
    folds_idx_oredered_by_score:list[int] = FOLDS_VAL_SCORE_ORDER.get(N_FOLDS, range(N_FOLDS))

    for fold_idx in folds_idx_oredered_by_score:
        yield *fold_indices[fold_idx], SEED + fold_idx
        # train_idx, val_idx = fold_indices[fold_idx]
        # yield Subset(dataset, train_idx), Subset(dataset, val_idx), SEED + fold_idx

# %%
def train_on_all_folds(
        # full_dataset: Dataset,
        lr_scheduler_kw: dict,
        optimizer_kw: dict,
        training_kw: dict,
        trial: Optional[optuna.trial.Trial]=None,
    ) -> tuple[float, DF, DF]:
    from time import time
    start_time = time()
    seed_everything(seed=SEED)
    ctx = mp.get_context("spawn")
    processes = []
    gpus = list(range(torch.cuda.device_count()))
    full_datasets = []
    for gpu_idx in gpus:
        device = torch.device(f"cuda:{gpu_idx}")
        full_datasets.append(CMIDataset(device))
    folds_it = sgkf_from_tensor_dataset(N_FOLDS)
    for fold_idx, (train_idx, validation_idx, seed) in enumerate(folds_it):
        gpu_idx = gpus[fold_idx % len(gpus)]
        p = ctx.Process(target=train_on_single_fold,
            args=(
                fold_idx,
                full_datasets[gpu_idx],
                train_idx,
                validation_idx,
                lr_scheduler_kw,
                optimizer_kw,
                training_kw,
                seed,
                gpu_idx,
            )
        )
        print("starting process for fold", fold_idx)
        p.start()
        processes.append(p)
    for p_idx, p in enumerate(processes):
        p.join()
        print("joined", p_idx)
        # all_epoch_metrics = pd.concat((all_epoch_metrics, epoch_metrics))
        # all_seq_meta_data_metrics = pd.concat((all_seq_meta_data_metrics, fold_seq_meta_data_metrics.assign(fold=fold_idx)))

        # if trial is not None:
        #     best_epoch_metrics = epoch_metrics.loc[epoch_metrics["final_metric"].idxmax()]
        #     trial.report(best_epoch_metrics["final_metric"], step=fold_idx)
        #     if trial.should_prune() and fold_idx < N_FOLDS - 1:
        #         print("Raising trial pruned exception.")
        #         raise TrialPruned()
    end_time = time()
    print(f"done in {end_time - start_time}s.")
    
    # print("\n" + "="*50)
    # print("Cross-Validation Results")
    # print("="*50)

    # # Statistiques pour les meilleures métriques
    # best_metrics:DF = (
    #     all_epoch_metrics
    #     .loc[:, ["binary_f1", "macro_f1", "final_metric"]]
    #     .groupby(level=0)
    #     .max()
    # )

    # print("\nBest Fold-wise Metrics:")
    # display(best_metrics)
    # print("\nGlobal Statistics (Best Metrics):")
    # print(f"Mean Best Final Metric: {best_metrics['final_metric'].mean():.4f} ± {best_metrics['final_metric'].std():.4f}")
    # print(f"Mean Best Binary F1: {best_metrics['binary_f1'].mean():.4f} ± {best_metrics['binary_f1'].std():.4f}")
    # print(f"Mean Best Macro F1: {best_metrics['macro_f1'].mean():.4f} ± {best_metrics['macro_f1'].std():.4f}")

    # return best_metrics["final_metric"].mean(), all_epoch_metrics, all_seq_meta_data_metrics
def train_on_all_folds(
        lr_scheduler_kw: dict,
        optimizer_kw: dict,
        training_kw: dict,
        trial: Optional[optuna.trial.Trial]=None,
    ) -> None:
    from time import time
    start_time = time()
    seed_everything(seed=SEED)
    ctx = mp.get_context("spawn")
    gpus = list(range(torch.cuda.device_count()))
    full_datasets = {gpu_idx: CMIDataset(torch.device(f"cuda:{gpu_idx}")) for gpu_idx in gpus}

    folds_it = list(sgkf_from_tensor_dataset(N_FOLDS))
    processes: list[mp.Process] = []

    # keep track of which GPU is free
    active: dict[int, mp.Process] = {}

    for fold_idx, (train_idx, validation_idx, seed) in enumerate(folds_it):
        # wait until a GPU is free
        while True:
            free_gpus = [gpu for gpu in gpus if gpu not in active]
            if free_gpus:
                gpu_idx = free_gpus[0]
                break
            # block until one process finishes
            for gpu_idx, proc in list(active.items()):
                proc.join(timeout=0.1)
                if not proc.is_alive():
                    proc.close()
                    del active[gpu_idx]

        # now gpu_idx is free
        p = ctx.Process(
            target=train_on_single_fold,
            args=(
                fold_idx,
                full_datasets[gpu_idx],
                train_idx,
                validation_idx,
                lr_scheduler_kw,
                optimizer_kw,
                training_kw,
                seed,
                gpu_idx,
            )
        )
        print(f"Starting process for fold {fold_idx} on GPU {gpu_idx}")
        p.start()
        active[gpu_idx] = p
        processes.append(p)

    # wait for all remaining processes
    for gpu_idx, proc in active.items():
        proc.join()
        proc.close()

    end_time = time()
    print(f"done in {end_time - start_time:.2f}s.")

# %%
if not os.getenv('KAGGLE_IS_COMPETITION_RERUN') and __name__ == "__main__":
    # full_dataset = CMIDataset()
    train_on_all_folds(
        # full_dataset,
        lr_scheduler_kw={
            'warmup_epochs': 14,
            'cycle_mult': 1.0,
            'max_lr': 0.00792127195137508,
            'init_cycle_epochs': 4,
            'lr_cycle_factor': 0.6,
            'max_to_min_div_factor': 250,
        },
        optimizer_kw={
            'weight_decay': 0.0009610813976803525, 
            'beta_0': 0.89889010289165792,
            'beta_1': 0.99722853486503933,
        },
        training_kw={
            'orient_loss_weight': 0.30000000000000004,
            'demos_loss_weight': 0.30000000000000004,
        },
    )
    # seq_meta_data_metrics.to_parquet("seq_meta_data_metrics.parquet")
    # user_input = input("Upload model ensemble?").lower()
    # if user_input == "yes":
    #     kagglehub.model_upload(
    #         handle=join(
    #             kagglehub.whoami()["username"],
    #             MODEL_NAME,
    #             "pyTorch",
    #             MODEL_VARIATION,
    #         ),
    #         local_model_dir="models",
    #         version_notes=input("Please provide model version notes:")
    #     )
    # elif user_input == "no":
    #     print("Model has not been uploaded to kaggle.")
    # else:
    #     print("User input was not understood, model has not been uploaded to kaggle.")

# %% [markdown]
# ## Hyperparameter tuning

# %%
class FoldPruner(BasePruner):
    def __init__(
        self,
        warmup_steps: int = 0,
        tolerance: float = 0.0,
    ):
        self.warmup_steps = int(warmup_steps)
        self.tolerance = float(tolerance)

    def prune(self, study: optuna.study.Study, current_trial: optuna.trial.FrozenTrial) -> bool:
        nb_trials = len(study.trials)
        if nb_trials < 2 or nb_trials < self.warmup_steps:
            return False
        
        current_fold = max(current_trial.intermediate_values.keys())
        # collect other trials' values at the same fold
        other_vals = []
        for t in study.trials[:-1]:
            if current_fold in t.intermediate_values:
                other_vals.append(t.intermediate_values[current_fold])
        # compute reference value from other_vals
        ref = max(other_vals)
        current_val = current_trial.intermediate_values[current_fold]
        
        return current_val < (ref - self.tolerance)

# %%
def objective(full_dataset:Dataset, trial: optuna.trial.Trial) -> float:
    return train_on_all_folds(
        full_dataset,
        lr_scheduler_kw={
            "warmup_epochs": trial.suggest_int("warmup_epochs", 12, 15),
            "cycle_mult": trial.suggest_float("cycle_mult", 0.9, 1.6, step=0.1),
            "max_lr": trial.suggest_float("max_lr", 0.005581907927062619 / 1.5, 0.005581907927062619 * 1.5, step=0.0001),
            "max_to_min_div_factor": 250, #trial.suggest_float("max_to_min_div_factor", 100, 300, step=25),
            "init_cycle_epochs": trial.suggest_int("init_cycle_epochs", 2, 10, ),
            "lr_cycle_factor": trial.suggest_float("lr_cycle_factor", 0.25, 0.6, step=0.05),
        },
        optimizer_kw={
            "weight_decay": trial.suggest_float("weight_decay", 5e-4, 1e-3),
            "beta_0":trial.suggest_float("beta_0", 0.8, 0.999),
            "beta_1":trial.suggest_float("beta_1", 0.99, 0.9999),
        },
        training_kw={
            "orient_loss_weight": trial.suggest_float("orient_loss_weight", 0, 1, step=0.1),
            "demos_loss_weight": trial.suggest_float("demos_loss_weight", 0, 1, step=0.1),
        },
        trial=trial,
    )[0]

# %%
# study = optuna.create_study(direction="maximize", pruner=FoldPruner(warmup_steps=0, tolerance=0.002))
# full_dataset = CMIDataset()
# study.optimize(partial(objective, full_dataset), n_trials=100, timeout=60 * 60 )

# pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
# complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

# print("Study statistics: ")
# print("  Number of finished trials: ", len(study.trials))
# print("  Number of pruned trials: ", len(pruned_trials))
# print("  Number of complete trials: ", len(complete_trials))
# print("Best trial:")
# current_trial = study.best_trial

# print("  Value: ", current_trial.value)

# print("  Params: ")
# for key, value in current_trial.params.items():
#     print("    {}: {}".format(key, value))

# %% [markdown]
# ## Submission

# %% [markdown]
# ### Reloading best models

# %%
def load_model_ensemble(parent_dir:str) -> list[nn.Module]:
    device = torch.device("cuda")
    model_ensemble = []
    for fold in range(N_FOLDS):
        model = mk_model()
        checkpoint = torch.load(
            join(
                parent_dir,
                f"model_fold_{fold}.pth"
            ),
            map_location=device,
            weights_only=True
        )
        model.load_state_dict(checkpoint)
        model.eval()
        model_ensemble.append(model)
    
    return model_ensemble
    
if __name__ == "__main__":
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        model_ensemble = load_model_ensemble("models")
    else:
        models_dir = kagglehub.model_download(
            join(
                KAGGLE_USERNAME,
                MODEL_NAME,
                "pyTorch",
                MODEL_VARIATION,
            )
        )
        model_ensemble = load_model_ensemble(models_dir)

# %% [markdown]
# ### Define prediction function

# %%
def preprocess_sequence_at_inference(sequence_df:pl.DataFrame) -> ndarray:
    return (
        sequence_df                     
        .to_pandas()                            # Convert to pandas dataframe.
        .pipe(imputed_features)                 # Impute missing data.
        .pipe(standardize_tof_cols_names)
        .pipe(norm_quat_rotations)              # Norm quaternions
        .pipe(add_linear_acc_cols)              # Add gravity free acceleration.
        .pipe(add_acc_magnitude, RAW_ACCELRATION_COLS, "acc_mag")
        .pipe(add_acc_magnitude, LINEAR_ACC_COLS, "linear_acc_mag")
        .pipe(add_quat_angle_mag)
        .pipe(add_angular_velocity_features)
        .pipe(rot_euler_angles)                 # Add rotation acc expressed as euler angles.
        .pipe(agg_tof_cols_per_sensor)          # Aggregate ToF columns.
        .pipe(add_diff_features)                # 
        .loc[:, sorted(meta_data["feature_cols"])]      # Retain only the usefull columns a.k.a features.
        # .sub(meta_data["mean"])                 # Subtract features by their mean, std norm pt.1.
        # .div(meta_data["std"])                  # Divide by Standard deviation, std norm pt.2.
        .pipe(length_normed_sequence_feat_arr, meta_data["pad_seq_len"], SEQ_PAD_TRUNC_MODE)  # get feature ndarray of sequence.
        .T                                      # Transpose to swap channel and X dimensions.
    )

def predict(sequence: pl.DataFrame, _: pl.DataFrame) -> str:
    """
    Kaggle evaluation API will call this for each sequence.
    sequence: polars DataFrame for a single sequence
    demographics: unused in this model
    Returns: predicted gesture string
    """
    x_tensor = (
        torch.unsqueeze(Tensor(preprocess_sequence_at_inference(sequence)), dim=0)
        .float()
        .cuda() #to(device)
    )
    print(x_tensor.shape)

    all_outputs = []
    with torch.no_grad():
        for model_idx, model in enumerate(model_ensemble): # Only take the first one bc it's the only one that takes in the correct input shape
            outputs, *_ = model(x_tensor)
            all_outputs.append(outputs)

    avg_outputs = torch.mean(torch.stack(all_outputs), dim=0)
    pred_idx = torch.argmax(avg_outputs, dim=1).item()

    return str(TARGET_NAMES[pred_idx])

# %% [markdown]
# ### Run inference server

# %%
if __name__ == "__main__":
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        competition_dataset_path = kagglehub.competition_download(COMPETITION_HANDLE)
        inference_server.run_local_gateway(
            data_paths=(
                join(competition_dataset_path, 'test.csv'),
                join(competition_dataset_path, 'test_demographics.csv'),
            )
        )
        inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)


