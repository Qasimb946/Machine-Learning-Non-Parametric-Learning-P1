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


class KNNRegressor:

    def __init__(self, k=3, gamma=1.0):
        self.k = k
        self.gamma = gamma
        self.X_train_ = None
        self.y_train_ = None

    def fit(self, X_train, y_train):
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float)

        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be 2D and y must be 1D.")

        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        if not isinstance(self.k, (int, np.integer)) or not 1 <= self.k <= len(X):
            raise ValueError("k must be an integer from 1 to the training size.")

        if not np.isfinite(self.gamma) or self.gamma < 0:
            raise ValueError("gamma must be finite and nonnegative.")

        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Training data must contain finite numbers.")

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        return self

    def predict(self, X_test):
        if self.X_train_ is None:
            raise ValueError("Call fit() before predict().")

        X_test = np.asarray(X_test, dtype=float)

        if X_test.ndim != 2:
            raise ValueError("X_test must be 2D.")

        if X_test.shape[1] != self.X_train_.shape[1]:
            raise ValueError("Training and test feature counts must match.")

        if not np.isfinite(X_test).all():
            raise ValueError("Test data must contain finite numbers.")

        predictions = []

        for query in X_test:
            distances = np.sqrt(
                np.sum((self.X_train_ - query) ** 2, axis=1)
            )

            nearest = np.argsort(
                distances, kind="stable"
            )[:self.k]

            neighbor_distances = distances[nearest]
            neighbor_targets = self.y_train_[nearest]

            # Stable weights: preserves the weighted-average prediction
            weights = np.exp(
                -self.gamma
                * (neighbor_distances - neighbor_distances.min())
            )

            prediction = (
                np.sum(weights * neighbor_targets) / weights.sum()
            )

            predictions.append(prediction)

        return np.asarray(predictions, dtype=float)


class EditedKNNRegressor(KNNRegressor):

    def __init__(self, k=3, gamma=1.0, epsilon=1.0, max_passes=10):
        super().__init__(k=k, gamma=gamma)

        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative.")

        if not isinstance(max_passes, (int, np.integer)) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer.")

        self.epsilon = epsilon
        self.max_passes = max_passes

    def fit(self, X_train, y_train):
        # Validate and store data using the parent class
        super().fit(X_train, y_train)

        X = self.X_train_
        y = self.y_train_
        kept = np.arange(len(y))

        self.reduction_passes_ = 0
        self.reduction_stop_ = "pass_limit"

        for pass_number in range(1, self.max_passes + 1):

            if len(kept) < 2:
                self.reduction_stop_ = "too_few_samples"
                break

            current_X = X[kept]
            current_y = y[kept]
            acceptable = np.empty(len(kept), dtype=bool)

            for i, query in enumerate(current_X):
                distances = np.sqrt(
                    np.sum((current_X - query) ** 2, axis=1)
                )

                # Exclude the sample itself
                distances[i] = np.inf
                nearest = np.argmin(distances)

                error = abs(current_y[i] - current_y[nearest])
                acceptable[i] = error <= self.epsilon

            self.reduction_passes_ = pass_number

            if acceptable.all():
                self.reduction_stop_ = "stable"
                break

            if acceptable.sum() < 2:
                self.reduction_stop_ = "minimum_size_guard"
                break

            # Remove samples simultaneously after each complete pass
            kept = kept[acceptable]

        self.retained_indices_ = kept
        self.X_train_ = X[kept].copy()
        self.y_train_ = y[kept].copy()
        self.effective_k_ = min(self.k, len(kept))

        return self

    def predict(self, X_test):
        # Parent predict() uses self.k, so temporarily apply the cap
        if self.X_train_ is None:
            raise ValueError("Call fit() before predict().")

        requested_k = self.k
        try:
            self.k = self.effective_k_
            return super().predict(X_test)
        finally:
            self.k = requested_k



class CondensedKNNRegressor(KNNRegressor):

    def __init__(self, k=3, gamma=1.0, epsilon=1.0,
                 random_state=42):
        super().__init__(k=k, gamma=gamma)

        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative.")

        self.epsilon = epsilon
        self.random_state = random_state

    def fit(self, X_train, y_train):
        # Validate and store the full training data
        super().fit(X_train, y_train)

        X = self.X_train_
        y = self.y_train_

        rng = np.random.default_rng(self.random_state)
        order = rng.permutation(len(y))

        # Start with one randomly selected sample
        kept = [int(order[0])]
        selected = np.zeros(len(y), dtype=bool)
        selected[kept[0]] = True

        self.reduction_passes_ = 0

        while True:
            self.reduction_passes_ += 1
            additions = 0

            for i in order:
                if selected[i]:
                    continue

                # Find the closest sample in the current retained set
                distances = np.sqrt(
                    np.sum((X[kept] - X[i]) ** 2, axis=1)
                )

                nearest_index = kept[np.argmin(distances)]
                error = abs(y[i] - y[nearest_index])

                if error > self.epsilon:
                    kept.append(int(i))
                    selected[i] = True
                    additions += 1

            # Stop when a complete pass adds nothing
            if additions == 0:
                break

        self.retained_indices_ = np.asarray(kept, dtype=int)
        self.X_train_ = X[kept].copy()
        self.y_train_ = y[kept].copy()
        self.effective_k_ = min(self.k, len(kept))
        self.reduction_stop_ = "stable"

        return self

    def predict(self, X_test):
        if self.X_train_ is None:
            raise ValueError("Call fit() before predict().")

        requested_k = self.k
        try:
            self.k = self.effective_k_
            return super().predict(X_test)
        finally:
            self.k = requested_k


