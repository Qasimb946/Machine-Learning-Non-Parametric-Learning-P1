"""
This file includes 5 × 2 cross-validation
"""
import numpy as np


def train_test_split(X, y):
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


def cross_validation(model, X, y, error_function):
    errors = []
    for i in range(5):
        X_A, X_B, y_A, y_B = train_test_split(X, y)

        model.fit(X_A, y_A)
        y_pred = model.predict(X_B)
        errors.append(error_function(y_B, y_pred))

        model.fit(X_B, y_B)
        y_pred = model.predict(X_A)
        errors.append(error_function(y_A, y_pred))

    return errors
