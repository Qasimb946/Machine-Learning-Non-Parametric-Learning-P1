"""
This file includes 5 × 2 cross-validation
"""
import numpy as np
import data_utils as du


def train_test_split(X, y):
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    midpoint = len(indices) // 2
    A_indices = indices[:midpoint]
    B_indices = indices[midpoint:]

    X_A = X[A_indices]
    X_B = X[B_indices]
    y_A = y[A_indices]
    y_B = y[B_indices]

    return X_A, X_B, y_A, y_B


def classification_error(y_true, y_pred):
    incorrect = np.sum(y_true != y_pred)
    total = len(y_true)
    return incorrect/total


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def cross_validation(model, X, y, error_function, seed=None):
    if seed is not None:
        np.random.seed(seed)
    errors = []
    for _ in range(5):
        X_A, X_B, y_A, y_B = train_test_split(X, y)

        X_min, X_max, X_A_norm = du.min_max_normalization(X_A)
        X_B_norm = du.helper_min_max_normalization(X_B, X_min, X_max)

        model.fit(X_A_norm, y_A)
        y_pred = model.predict(X_B_norm)
        errors.append(error_function(y_B, y_pred))

        X_min, X_max, X_B_norm = du.min_max_normalization(X_B)
        X_A_norm = du.helper_min_max_normalization(X_A, X_min, X_max)

        model.fit(X_B_norm, y_B)
        y_pred = model.predict(X_A_norm)
        errors.append(error_function(y_A, y_pred))

    return errors
