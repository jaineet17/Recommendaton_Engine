"""
Common utilities for the recommendation engine (Compatibility Module).

This module has been deprecated and consolidated into utility.py.
It is maintained for backward compatibility only.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Issue deprecation warning
warnings.warn(
    "This module has been consolidated into utility.py. "
    "Please update your imports to use 'from src.utils.utility import *' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new location
from src.utils.utility import (
    load_pickle, save_pickle, normalize_vector, cosine_similarity,
    calculate_precision_at_k, calculate_recall_at_k, calculate_ndcg_at_k,
    get_top_n_indices, Timer, log_exception, create_audit_log
)

# Re-export the logger for backward compatibility
from src.utils.logging_config import get_logger
logger = get_logger('utils.common')

# The functions below are now imported from utility.py
# This module is kept for backward compatibility only
