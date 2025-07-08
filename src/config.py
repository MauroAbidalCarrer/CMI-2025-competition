COMPETITION_HANDLE = "cmi-detect-behavior-with-sensor-data"
# Dataset
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
DATASET_DF_DTYPES = {col: "category" for col in CATEGORY_COLUMNS}
SAMPLING_FREQUENCY = 10 #Hz