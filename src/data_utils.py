"""
This file includes data preprocessing
"""
import numpy as np


def load_data(file_path, has_header=False, missing_value=None):
    header = 1 if has_header else 0
    data = np.genfromtxt(file_path, delimiter=",", dtype=str, skip_header=header)

    if missing_value is not None:
        rows_missing = np.any(data == missing_value, axis=1)
        data = data[~rows_missing]
    return data


def min_max_normalization(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X-X_min)/(X_max - X_min)
