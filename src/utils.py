import os
import random

import torch
import numpy as np

from config import *


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

def ignore_warnings():
    warnings.filterwarnings(
        "ignore",
        message=(
            "DataFrame is highly fragmented.  This is usually the result of "
            "calling `frame.insert` many times.*"
        ),
        category=pd.errors.PerformanceWarning,
    )
    warnings.filterwarnings("ignore", message=".*sm_120.*")
    warnings.filterwarnings("ignore", message="The distribution is specified*")
