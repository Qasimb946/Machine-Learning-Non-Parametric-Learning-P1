"""
This file includes all regression models
"""

import numpy as np
import pandas as pd


class NullRegressor:

    def __init__(self):
        self.mean_ = None

    def fit(self, X_train, y_train):
        self.mean_ = np.asarray(y_train, dtype=float).mean()
        return self

    def predict(self, X_test):
        if self.mean_ is None:
            raise ValueError("Call fit() before predict().")

        return np.full(len(X_test), self.mean_)
