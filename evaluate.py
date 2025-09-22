# evaluate.py
# Time-series evaluation utilities for point forecasts, intervals, and backtests.
# No hard dependency on your model files—pass callables or arrays.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import math
import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series, list]


# ---------------------------
# Basic metric implementations
# ---------------------------

def _to_1d(a: ArrayLike) -> np.ndarray:
    arr = np.asarray(a, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Empty array provided.")
    return arr


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, yhat = _to_1d(y_true), _to_1d(y_pred)
    _check_equal_length(y, yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y, yhat = _to_1d(y_true), _to_1d(y_pred)
    _check_equal_length(y, yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-8) -> float:
    """
