from typing import Literal
from itertools import product

import numpy as np

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
N_TARGETS = len(TARGET_NAMES)
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
NON_BFRB_GESTURES = list(set(TARGET_NAMES) - set(BFRB_GESTURES))
TARGET_NAMES = BFRB_GESTURES + NON_BFRB_GESTURES
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
GRAVITY_COLS = ["gravity_x", "gravity_y", "gravity_z"]
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
N_TOF_SENSORS = 5
N_THM_SENSORS = 5
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
BINARY_DEMOS_TARGETS = ["sex", "handedness"]
REGRES_DEMOS_TARGETS = [
    # "age",
    "height_cm",
    # "shoulder_to_wrist_cm",
    # "elbow_to_wrist_cm",
    "arm_length_ratio",
    "elbow_to_wrist_ratio",
    "shoulder_to_elbow_ratio",
]
# Data augmentation
JITTER = 0.25
SCALING = 0.2
MIXUP = 0.3
LABEL_SMOOTHING = 0.1
# Training loop
N_FOLDS = 24
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
# expert model
KAGGLE_USERNAME = "mauroabidalcarrer"
MODEL_NAME = "cmi-model"
MODEL_VARIATION = "single_model_architecture"
DFLT_MODEL_HP_KW = {
    "group_thm_branch": True,
    "rnn_gaussian_noise": 0.1,
    "imu_dropout_ratio": 0.2,
    "thm_dropout_ratio": 0.2,
    "tof_dropout_ratio": 0.2,
    "head_dropout_ratio": 0,
    "mlp_width": 256,
}
DFLT_LR_SCHEDULER_HP_KW={
    'warmup_epochs': 17,
    'cycle_mult': 1.3,
    'init_cycle_epochs': 6,
    'max_lr': 0.00832127195137508,
    'lr_cycle_factor': 0.5,
    'max_to_min_div_factor': 250,
}
DFLT_OPTIMIZER_HP_KW={
    'weight_decay': 0.0009212879238672411,
    'beta_0': 0.8161978952748745,
    'beta_1': 0.9935729096966865,
}
DFLT_TRAINING_HP_KW={
    'orient_loss_weight': 0.15000000000000005,
    'sex_loss_weight': 0.1,
    'handedness_loss_weight': 0.0, 
    "arm_length_ratio_loss_weight": 0.0,
    "elbow_to_wrist_ratio_loss_weight": 0.0,
    "shoulder_to_elbow_ratio_loss_weight": 0.0,
    "height_cm_loss_weight": 0.0,
    "age_loss_weight": 0,
    "mixup_alpha": 0.2,
    "mixup_ratio": 1,
    "focal_gamma": 1,
    "model_ema_decay": 0.8,
}
# gating model
GATING_INPUT_FEATURES = [
    "bin_mae",
    "reg_mae",
    "y_uncertainty",
    "bin_uncertainty",
    "reg_uncertainty",
    "orient_uncertainty",
]
GATING_MODEL_TRAIN_BATCH_SIZE = 256
GATING_MODEL_TEST_BATCH_SIZE = GATING_MODEL_TRAIN_BATCH_SIZE * 4
N_GATING_MODEL_EPOCHS = 5
