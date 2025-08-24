import re
import os
import json 
import shutil
from os.path import join
from functools import partial
from tqdm import tqdm
from operator import methodcaller
from typing import Literal

import kagglehub 
import numpy as np
import pandas as pd
from numpy import ndarray
from numpy.linalg import norm
from pandas import DataFrame as DF
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader as DL
from numpy.lib.stride_tricks import sliding_window_view

from config import *
from utils import seed_everything


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
        "preprocessed_dataset/binary_demographics_Y.npy",
        seq_meta_data[BINARY_DEMOS_TARGETS].values.astype("float32"),
    )
    np.save(
        "preprocessed_dataset/regres_demographics_Y.npy",
        seq_meta_data[REGRES_DEMOS_TARGETS].values.astype("float32"),
    )

# Convert target names into a ndarray to index it batchwise.
def get_sensor_indices(sensor_prefix: str, meta_data: dict) -> list[int]:
    is_sensor_feat = methodcaller("startswith", sensor_prefix)
    return [feat_idx for feat_idx, feat in enumerate(meta_data["feature_cols"]) if is_sensor_feat(feat)]

def save_df_meta_data(df:DF):
    meta_data = {
        "mean": df[get_feature_cols(df)].mean().astype("float32").to_dict(),
        "std": df[get_feature_cols(df)].std().astype("float32").to_dict(),
        "pad_seq_len": get_normed_seq_len(df),
        "feature_cols": get_feature_cols(df),
        "n_orient_classes": df["orientation"].nunique(),
    }
    meta_data["tof_idx"] = get_sensor_indices("tof", meta_data)
    meta_data["thm_idx"] = get_sensor_indices("thm", meta_data)
    meta_data["imu_idx"] = list(filter(lambda idx: idx not in meta_data["tof_idx"] + meta_data["thm_idx"], range(len(meta_data["feature_cols"]))))

    with open("preprocessed_dataset/full_dataset_meta_data.json", "w") as fp:
        json.dump(meta_data, fp, indent=4)

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

def get_meta_data() -> dict:
    meta_data_path = join(
        "preprocessed_dataset",
        "full_dataset_meta_data.json"
    )
    with open(meta_data_path, "r") as fp:
        meta_data = json.load(fp)

    return meta_data


if __name__ == "__main__":
    seed_everything(SEED)
    create_preprocessed_dataset()