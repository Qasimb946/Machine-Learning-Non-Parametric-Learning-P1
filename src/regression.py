"""
This file includes all regression models
"""

import numpy as np
import pandas as pd


def _combined_squared_distances(X_train, query,
                                 X_cyclic_train=None, query_cyclic=None,
                                 cyclic_periods=None):

    main_sq = np.sum((X_train - query) ** 2, axis=1)

    if cyclic_periods is None or X_cyclic_train is None or query_cyclic is None:
        return main_sq

    cyclic_sq = np.zeros(len(X_train))
    for j, period in enumerate(cyclic_periods):
        raw_diff = np.abs(X_cyclic_train[:, j] - query_cyclic[j])
        circ = np.minimum(raw_diff, period - raw_diff)
        normalized = circ / (period / 2.0)
        cyclic_sq += normalized ** 2

    return main_sq + cyclic_sq


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

    def __init__(self, k=3, gamma=1.0, cyclic_periods=None):
        self.k = k
        self.gamma = gamma
        # e.g. cyclic_periods=[12, 7] for [month, day]. None => no cyclic
        # columns at all; existing (Abalone) callers are unaffected.
        self.cyclic_periods = cyclic_periods
        self.X_train_ = None
        self.y_train_ = None
        self.X_cyclic_train_ = None

    def fit(self, X_train, y_train, X_cyclic_train=None):
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

        if self.cyclic_periods is not None:
            if X_cyclic_train is None:
                raise ValueError("cyclic_periods was set but no X_cyclic_train given.")
            X_cyclic_train = np.asarray(X_cyclic_train, dtype=float)
            if len(X_cyclic_train) != len(X):
                raise ValueError("X_cyclic_train must have the same number of rows as X_train.")
            if X_cyclic_train.shape[1] != len(self.cyclic_periods):
                raise ValueError("X_cyclic_train column count must match len(cyclic_periods).")
            self.X_cyclic_train_ = X_cyclic_train.copy()

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        return self

    def predict(self, X_test, X_cyclic_test=None):
        if self.X_train_ is None:
            raise ValueError("Call fit() before predict().")

        X_test = np.asarray(X_test, dtype=float)

        if X_test.ndim != 2:
            raise ValueError("X_test must be 2D.")

        if X_test.shape[1] != self.X_train_.shape[1]:
            raise ValueError("Training and test feature counts must match.")

        if not np.isfinite(X_test).all():
            raise ValueError("Test data must contain finite numbers.")

        if self.cyclic_periods is not None:
            if X_cyclic_test is None:
                raise ValueError("cyclic_periods was set but no X_cyclic_test given.")
            X_cyclic_test = np.asarray(X_cyclic_test, dtype=float)
            if len(X_cyclic_test) != len(X_test):
                raise ValueError("X_cyclic_test must have the same number of rows as X_test.")

        predictions = []

        for idx, query in enumerate(X_test):
            query_cyclic = X_cyclic_test[idx] if X_cyclic_test is not None else None

            sq_distances = _combined_squared_distances(
                self.X_train_, query,
                self.X_cyclic_train_, query_cyclic,
                self.cyclic_periods
            )
            distances = np.sqrt(sq_distances)

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

    def __init__(self, k=3, gamma=1.0, epsilon=1.0, max_passes=10,
                 cyclic_periods=None):
        super().__init__(k=k, gamma=gamma, cyclic_periods=cyclic_periods)

        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative.")

        if not isinstance(max_passes, (int, np.integer)) or max_passes < 1:
            raise ValueError("max_passes must be a positive integer.")

        self.epsilon = epsilon
        self.max_passes = max_passes

    def fit(self, X_train, y_train, X_cyclic_train=None):
        # Validate and store data using the parent class
        super().fit(X_train, y_train, X_cyclic_train=X_cyclic_train)

        X = self.X_train_
        y = self.y_train_
        X_cyc = self.X_cyclic_train_  # None if no cyclic columns
        kept = np.arange(len(y))

        self.reduction_passes_ = 0
        self.reduction_stop_ = "pass_limit"

        for pass_number in range(1, self.max_passes + 1):

            if len(kept) < 2:
                self.reduction_stop_ = "too_few_samples"
                break

            current_X = X[kept]
            current_y = y[kept]
            current_X_cyc = X_cyc[kept] if X_cyc is not None else None
            acceptable = np.empty(len(kept), dtype=bool)

            for i, query in enumerate(current_X):
                query_cyclic = current_X_cyc[i] if current_X_cyc is not None else None

                # Editing only needs to know WHO is nearest (a ranking
                # question), so squared distance is enough -- no sqrt
                # needed here, same reasoning as classification's
                # unrooted Dnum+Dcat: argmin is unaffected by a
                # monotonic transform.
                sq_distances = _combined_squared_distances(
                    current_X, query, current_X_cyc, query_cyclic,
                    self.cyclic_periods
                )

                # Exclude the sample itself
                sq_distances[i] = np.inf
                nearest = np.argmin(sq_distances)

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
        if X_cyc is not None:
            self.X_cyclic_train_ = X_cyc[kept].copy()
        self.effective_k_ = min(self.k, len(kept))

        return self

    def predict(self, X_test, X_cyclic_test=None):
        # Parent predict() uses self.k, so temporarily apply the cap
        if self.X_train_ is None:
            raise ValueError("Call fit() before predict().")

        requested_k = self.k
        try:
            self.k = self.effective_k_
            return super().predict(X_test, X_cyclic_test=X_cyclic_test)
        finally:
            self.k = requested_k



class CondensedKNNRegressor(KNNRegressor):

    def __init__(self, k=3, gamma=1.0, epsilon=1.0,
                 random_state=42, cyclic_periods=None):
        super().__init__(k=k, gamma=gamma, cyclic_periods=cyclic_periods)

        if not np.isfinite(epsilon) or epsilon < 0:
            raise ValueError("epsilon must be finite and nonnegative.")

        self.epsilon = epsilon
        self.random_state = random_state

    def fit(self, X_train, y_train, X_cyclic_train=None):
        # Validate and store the full training data
        super().fit(X_train, y_train, X_cyclic_train=X_cyclic_train)

        X = self.X_train_
        y = self.y_train_
        X_cyc = self.X_cyclic_train_  # None if no cyclic columns

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

                query_cyclic = X_cyc[i] if X_cyc is not None else None
                kept_X_cyc = X_cyc[kept] if X_cyc is not None else None

                # Same ranking-only argument as EditedKNNRegressor:
                # finding the closest retained sample only needs order,
                # so squared distance (no sqrt) is sufficient here.
                sq_distances = _combined_squared_distances(
                    X[kept], X[i], kept_X_cyc, query_cyclic,
                    self.cyclic_periods
                )

                nearest_index = kept[np.argmin(sq_distances)]
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
        if X_cyc is not None:
            self.X_cyclic_train_ = X_cyc[kept].copy()
        self.effective_k_ = min(self.k, len(kept))
        self.reduction_stop_ = "stable"

        return self

    def predict(self, X_test, X_cyclic_test=None):
        if self.X_train_ is None:
            raise ValueError("Call fit() before predict().")

        requested_k = self.k
        try:
            self.k = self.effective_k_
            return super().predict(X_test, X_cyclic_test=X_cyclic_test)
        finally:
            self.k = requested_k