from itertools import product

# Dataset
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
N_FOLDS = 5
VALIDATION_FRACTION = 0.2
