"""
This file includes data preprocessing
"""
import numpy as np
import pandas as pd


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
    return X_min, X_max, (X-X_min)/(X_max - X_min)

def helper_min_max_normalization(X, X_min, X_max):
    return (X-X_min)/(X_max - X_min)


import pandas as pd


class RegressionPreprocessor:

    def __init__(self, numerical_cols, categorical_levels=None):
        self.numerical_cols = list(numerical_cols)
        self.categorical_levels = categorical_levels or {}
        self.mean_ = None
        self.std_ = None

    def fit(self, X_train):
        numeric = X_train[self.numerical_cols].astype(float)

        self.mean_ = numeric.mean()
        self.std_ = numeric.std(ddof=0).replace(0, 1)

        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("Call fit() before transform().")

        numeric = X[self.numerical_cols].astype(float)

        result = (numeric - self.mean_) / self.std_

        for column, categories in self.categorical_levels.items():

            if not X[column].isin(categories).all():
                raise ValueError(
                    f"Missing or unknown category in {column}."
                )

            for category in categories:
                result[f"{column}_{category}"] = (
                    X[column] == category
                ).astype(float)

        values = result.to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError("Prepared features contain NaN or infinity.")

        return values

