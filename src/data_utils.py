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


def load_classification_dataset(dataset_name):
    if dataset_name == "breast":
        data = load_data(
            "data/breast-cancer-wisconsin.data",
            missing_value="?"
        )

        X = data[:, 1:-1].astype(float)
        y = data[:, -1]
        normalize = True

    elif dataset_name == "car":
        data = load_data("data/car.data")

        X = data[:, :-1]
        y = data[:, -1]

        X = one_hot_encoding(X)
        normalize = False

    elif dataset_name == "vote":
        data = load_data("data/house-votes-84.data")

        X = data[:, 1:]
        y = data[:, 0]

        X = one_hot_encoding(X)
        normalize = False

    else:
        raise ValueError("Unknown classification dataset.")

    return X, y, normalize


def min_max_normalization(X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return X_min, X_max, (X-X_min)/(X_max - X_min)


def helper_min_max_normalization(X, X_min, X_max):
    return (X-X_min)/(X_max - X_min)


def one_hot_encoding(X):
    encoded_columns = []
    for col in range(X.shape[1]):
        values = X[:, col]
        categories = np.unique(values)
        encoded = np.zeros((len(values), len(categories)))
        for i, value in enumerate(values):
            category_index = np.where(categories == value)[0][0]
            encoded[i, category_index] = 1
        encoded_columns.append(encoded)
    return np.hstack(encoded_columns)

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


MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
DAY_ORDER = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

CYCLIC_PERIODS = {"month": len(MONTH_ORDER), "day": len(DAY_ORDER)}


def encode_cyclic_columns(df, columns):

    orders = {"month": MONTH_ORDER, "day": DAY_ORDER}
    encoded_cols = []
    periods = []

    for col in columns:
        if col not in orders:
            raise ValueError(f"No cyclic ordering defined for column '{col}'.")

        order = orders[col]
        lookup = {name: i for i, name in enumerate(order)}

        values = df[col].astype(str).str.strip().str.lower().map(lookup)

        if values.isna().any():
            bad = df[col][values.isna()].unique()
            raise ValueError(f"Unrecognized value(s) in column '{col}': {bad}")

        encoded_cols.append(values.to_numpy(dtype=float))
        periods.append(len(order))

    encoded = np.column_stack(encoded_cols) if encoded_cols else np.empty((len(df), 0))
    return encoded, periods
