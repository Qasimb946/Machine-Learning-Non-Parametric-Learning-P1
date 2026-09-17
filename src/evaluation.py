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


def macro_average(y_true, y_pred):
    classes = np.unique(y_true)

    precisions = []
    recalls = []

    for c in classes:
        TP = np.sum((y_true == c) & (y_pred == c))
        FP = np.sum((y_true != c) & (y_pred == c))
        FN = np.sum((y_true == c) & (y_pred != c))

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)

    return macro_precision, macro_recall


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def tune_classifier(X_train, y_train, model_class, normalize=True, seed=42):
    train_idx, valid_idx = split_inner_indices(len(X_train), random_state=seed)

    X_inner = X_train[train_idx]
    y_inner = y_train[train_idx]

    X_valid = X_train[valid_idx]
    y_valid = y_train[valid_idx]

    if normalize:
        X_min, X_max, X_inner_ready = du.min_max_normalization(X_inner)
        X_valid_ready = du.helper_min_max_normalization(X_valid, X_min, X_max)
    else:
        X_inner_ready = X_inner
        X_valid_ready = X_valid

    results = []

    for p in [1, 2]:
        for k in range(1, 20):
            np.random.seed(seed)
            model = model_class(k=k, p=p)
            model.fit(X_inner_ready, y_inner)
            y_pred = model.predict(X_valid_ready)
            error = classification_error(y_valid, y_pred)
            results.append([k, p, error])

    results = np.array(results)

    best_index = np.argmin(results[:, 2])
    best_result = results[best_index]

    return int(best_result[0]), int(best_result[1]), best_result[2]


def tuned_cross_validation(model_class, X, y, error_function, seed=42, normalize=True):
    errors = []
    selected_params = []
    retained_percentages = []
    macro_scores = []

    np.random.seed(seed)

    for i in range(5):
        X_A, X_B, y_A, y_B = train_test_split(X, y)

        # A trains, B tests
        best_k, best_p, validation_error = tune_classifier(X_A, y_A, model_class, normalize=normalize, seed=seed + i)
        selected_params.append([best_k, best_p, validation_error])

        if normalize:
            X_min, X_max, X_A_ready = du.min_max_normalization(X_A)
            X_B_ready = du.helper_min_max_normalization(X_B, X_min, X_max)
        else:
            X_A_ready = X_A
            X_B_ready = X_B

        model = model_class(k=best_k, p=best_p)
        model.fit(X_A_ready, y_A)
        if hasattr(model, "X_train_sub"):
            retained_percentages.append(100 * len(model.X_train_sub) / len(X_A))
        else:
            retained_percentages.append(None)

        y_pred = model.predict(X_B_ready)
        errors.append(error_function(y_B, y_pred))

        macro_precision, macro_recall = macro_average(y_B, y_pred)
        macro_scores.append([macro_precision, macro_recall])

        # B trains, A tests
        best_k, best_p, validation_error = tune_classifier(X_B, y_B, model_class, normalize=normalize, seed=seed + i + 100)
        selected_params.append([best_k, best_p, validation_error])

        if normalize:
            X_min, X_max, X_B_ready = du.min_max_normalization(X_B)
            X_A_ready = du.helper_min_max_normalization(X_A, X_min, X_max)
        else:
            X_B_ready = X_B
            X_A_ready = X_A

        model = model_class(k=best_k, p=best_p)
        model.fit(X_B_ready, y_B)

        if hasattr(model, "X_train_sub"):
            retained_percentages.append(100 * len(model.X_train_sub) / len(X_B))
        else:
            retained_percentages.append(None)

        y_pred = model.predict(X_A_ready)
        errors.append(error_function(y_A, y_pred))

        macro_precision, macro_recall = macro_average(y_A, y_pred)
        macro_scores.append([macro_precision, macro_recall])

    return errors, np.array(selected_params), retained_percentages, np.array(macro_scores)


def cross_validation(model, X, y, error_function, seed=None, normalize=True):
    if seed is not None:
        np.random.seed(seed)
    errors = []

    for _ in range(5):
        X_A, X_B, y_A, y_B = train_test_split(X, y)

        if normalize:
            X_min, X_max, X_A_ready = du.min_max_normalization(X_A)
            X_B_ready = du.helper_min_max_normalization(X_B, X_min, X_max)
        else:
            X_A_ready = X_A
            X_B_ready = X_B

        model.fit(X_A_ready, y_A)
        y_pred = model.predict(X_B_ready)
        errors.append(error_function(y_B, y_pred))

        if normalize:
            X_min, X_max, X_B_ready = du.min_max_normalization(X_B)
            X_A_ready = du.helper_min_max_normalization(X_A, X_min, X_max)
        else:
            X_B_ready = X_B
            X_A_ready = X_A

        model.fit(X_B_ready, y_B)
        y_pred = model.predict(X_A_ready)
        errors.append(error_function(y_A, y_pred))

    return errors


def generate_5x2_indices(n_samples, random_state=42):
    """Yield training/testing row positions for all ten folds."""
    if n_samples < 2:
        raise ValueError("At least two samples are required.")

    rng = np.random.default_rng(random_state)

    for rep in range(5):
        positions = rng.permutation(n_samples)
        midpoint = n_samples // 2

        A = positions[:midpoint]
        B = positions[midpoint:]

        yield rep, 1, A, B
        yield rep, 2, B, A


def split_inner_indices(n_samples, validation_fraction=0.2,
                        random_state=42):
    """Split an outer training half into inner training/validation."""
    if n_samples < 2:
        raise ValueError("At least two samples are required.")

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    rng = np.random.default_rng(random_state)
    positions = rng.permutation(n_samples)

    split_position = int((1 - validation_fraction) * n_samples)

    if not 1 <= split_position < n_samples:
        raise ValueError("Both subsets must contain at least one sample.")

    return positions[:split_position], positions[split_position:]


from data_utils import RegressionPreprocessor
from regression import (
    KNNRegressor,
    EditedKNNRegressor,
    CondensedKNNRegressor,
)
import pandas as pd
from regression import NullRegressor


def tune_regressor(
    X, y, model_name,
    numerical_cols,
    categorical_levels,
    k_candidates,
    gamma_candidates,
    epsilon_candidates=None,
    random_state=42,
    cyclic_cols=None
):
    # X is a DataFrame containing inputs only
    # cyclic_cols: e.g. ["month", "day"] -- None means unaffected (Abalone).
    y = np.asarray(y, dtype=float)

    if y.ndim != 1 or len(X) != len(y):
        raise ValueError("y must be 1D and match the number of X rows.")

    model_classes = {
        "knn": KNNRegressor,
        "edited": EditedKNNRegressor,
        "condensed": CondensedKNNRegressor,
    }

    if model_name not in model_classes:
        raise ValueError("Use 'knn', 'edited', or 'condensed'.")

    train_idx, valid_idx = split_inner_indices(
        len(X), random_state=random_state
    )

    preprocessor = RegressionPreprocessor(
        numerical_cols=numerical_cols,
        categorical_levels=categorical_levels
    )

    preprocessor.fit(X.iloc[train_idx])

    X_inner = preprocessor.transform(X.iloc[train_idx])
    X_valid = preprocessor.transform(X.iloc[valid_idx])

    y_inner = y[train_idx]
    y_valid = y[valid_idx]

    cyclic_periods = None
    X_cyc_inner = None
    X_cyc_valid = None
    if cyclic_cols:
        X_cyc_inner, cyclic_periods = du.encode_cyclic_columns(
            X.iloc[train_idx], cyclic_cols
        )
        X_cyc_valid, _ = du.encode_cyclic_columns(
            X.iloc[valid_idx], cyclic_cols
        )

    k_values = list(k_candidates)
    gamma_values = list(gamma_candidates)

    if model_name == "knn":
        epsilon_values = [None]
    else:
        if epsilon_candidates is None:
            raise ValueError("Reduced models require epsilon candidates.")
        epsilon_values = list(epsilon_candidates)

    best_params = None
    best_key = None
    trials = []

    for epsilon in epsilon_values:

        if model_name == "knn":
            model = KNNRegressor(k=1, gamma=1.0, cyclic_periods=cyclic_periods)
        else:
            settings = dict(k=1, gamma=1.0, epsilon=epsilon,
                             cyclic_periods=cyclic_periods)

            if model_name == "condensed":
                settings["random_state"] = random_state

            model = model_classes[model_name](**settings)

        model.fit(X_inner, y_inner, X_cyclic_train=X_cyc_inner)
        retained_n = len(model.y_train_)

        for k in k_values:
            if not isinstance(k, (int, np.integer)) or k < 1:
                raise ValueError("k candidates must be positive integers.")

            if k > retained_n:
                continue

            for gamma in gamma_values:
                if not np.isfinite(gamma) or gamma < 0:
                    raise ValueError("gamma must be finite and nonnegative.")

                model.k = int(k)
                model.gamma = float(gamma)

                if model_name != "knn":
                    model.effective_k_ = int(k)

                predictions = model.predict(X_valid, X_cyclic_test=X_cyc_valid)
                score = float(mean_squared_error(y_valid, predictions))

                if not np.isfinite(score):
                    raise ValueError("Validation MSE is not finite.")

                params = {
                    "k": int(k),
                    "gamma": float(gamma)
                }

                if model_name != "knn":
                    params["epsilon"] = float(epsilon)

                trials.append({
                    **params,
                    "validation_mse": score,
                    "retained_n": retained_n
                })

                key = (
                    score, int(k), float(gamma),
                    0.0 if epsilon is None else float(epsilon)
                )

                if best_key is None or key < best_key:
                    best_key = key
                    best_params = params.copy()

    if best_params is None:
        raise ValueError("No valid parameter combinations were evaluated.")

    return best_params, best_key[0], trials


def run_regression_5x2(
    X, y,
    numerical_cols,
    categorical_levels,
    k_candidates,
    gamma_candidates,
    epsilon_candidates,
    random_state=42,
    cyclic_cols=None
):
    # cyclic_cols: e.g. ["month", "day"]. None => unchanged (Abalone).
    y = np.asarray(y, dtype=float)

    if y.ndim != 1 or len(X) != len(y):
        raise ValueError("y must be 1D and match the number of X rows.")

    k_candidates = list(k_candidates)
    gamma_candidates = list(gamma_candidates)
    epsilon_candidates = list(epsilon_candidates)

    model_classes = {
        "knn": KNNRegressor,
        "edited": EditedKNNRegressor,
        "condensed": CondensedKNNRegressor,
    }

    results = []
    tuning_results = []
    prediction_results = []

    for rep, fold, train_idx, test_idx in generate_5x2_indices(
        len(X), random_state=random_state
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        inner_seed = random_state + rep * 2 + fold
        selected = {}

        for name in model_classes:
            print(
                f"Repetition {rep + 1}, fold {fold}: tuning {name}",
                flush=True
            )

            params, validation_mse, trials = tune_regressor(
                X_train,
                y_train,
                model_name=name,
                numerical_cols=numerical_cols,
                categorical_levels=categorical_levels,
                k_candidates=k_candidates,
                gamma_candidates=gamma_candidates,
                epsilon_candidates=epsilon_candidates,
                random_state=inner_seed,
                cyclic_cols=cyclic_cols
            )

            selected[name] = (params, validation_mse)

            for trial in trials:
                tuning_results.append({
                    "repetition": rep + 1,
                    "fold": fold,
                    "model": name,
                    **trial
                })

        preprocessor = RegressionPreprocessor(
            numerical_cols=numerical_cols,
            categorical_levels=categorical_levels
        )

        preprocessor.fit(X_train)
        X_train_ready = preprocessor.transform(X_train)
        X_test_ready = preprocessor.transform(X_test)

        cyclic_periods = None
        X_cyc_train_ready = None
        X_cyc_test_ready = None
        if cyclic_cols:
            X_cyc_train_ready, cyclic_periods = du.encode_cyclic_columns(
                X_train, cyclic_cols
            )
            X_cyc_test_ready, _ = du.encode_cyclic_columns(
                X_test, cyclic_cols
            )

        for name in ["null", "knn", "edited", "condensed"]:

            if name == "null":
                params = {}
                validation_mse = None
                model = NullRegressor()
            else:
                params, validation_mse = selected[name]
                settings = params.copy()
                settings["cyclic_periods"] = cyclic_periods

                if name == "condensed":
                    settings["random_state"] = inner_seed

                model = model_classes[name](**settings)

            if name == "null":
                model.fit(X_train_ready, y_train)
                predictions = model.predict(X_test_ready)
            else:
                model.fit(X_train_ready, y_train, X_cyclic_train=X_cyc_train_ready)
                predictions = model.predict(X_test_ready, X_cyclic_test=X_cyc_test_ready)

            if predictions.shape != y_test.shape:
                raise ValueError("Prediction and target shapes differ.")

            test_mse = float(mean_squared_error(y_test, predictions))
            test_rmse = float(root_mean_squared_error(y_test, predictions))
            test_mae = float(mean_absolute_error(y_test, predictions))

            if not all(np.isfinite([test_mse, test_rmse, test_mae])):
                raise ValueError("One or more test metrics are not finite.")

            if name == "null":
                retained_n = None
                retained_percent = None
            else:
                retained_n = len(model.y_train_)
                retained_percent = 100 * retained_n / len(y_train)

            results.append({
                "repetition": rep + 1,
                "fold": fold,
                "model": name,
                "selected_k": params.get("k"),
                "effective_k": getattr(
                    model, "effective_k_", params.get("k")
                ),
                "gamma": params.get("gamma"),
                "epsilon": params.get("epsilon"),
                "validation_mse": validation_mse,
                "test_mse": test_mse,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "train_n": len(y_train),
                "test_n": len(y_test),
                "retained_n": retained_n,
                "retained_percent": retained_percent,
                "reduction_passes": getattr(
                    model, "reduction_passes_", 0
                ),
                "reduction_stop": getattr(
                    model, "reduction_stop_", "not_applicable"
                )
            })

            for position, actual, predicted in zip(
                test_idx, y_test, predictions
            ):
                prediction_results.append({
                    "repetition": rep + 1,
                    "fold": fold,
                    "model": name,
                    "row_position": int(position),
                    "actual": float(actual),
                    "prediction": float(predicted)
                })

        print(
            f"Completed repetition {rep + 1}, fold {fold}",
            flush=True
        )

    results_df = pd.DataFrame(results)
    tuning_df = pd.DataFrame(tuning_results)
    predictions_df = pd.DataFrame(prediction_results)

    summary_df = results_df.groupby("model").agg(
        mean_test_mse=("test_mse", "mean"),
        std_fold_mse=("test_mse", "std"),
        mean_test_rmse=("test_rmse", "mean"),
        std_fold_rmse=("test_rmse", "std"),
        mean_test_mae=("test_mae", "mean"),
        std_fold_mae=("test_mae", "std"),
        mean_retained_percent=("retained_percent", "mean")
        ).reset_index()

    return results_df, summary_df, tuning_df, predictions_df