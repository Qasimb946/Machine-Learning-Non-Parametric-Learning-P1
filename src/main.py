"""
This file is the driver for project 1
"""
import numpy as np
import data_utils as du
import classification as cls
import evaluation as evl

import pandas as pd
from pathlib import Path

def run_hardware_experiment(log_transform=True):
    project_dir = Path(__file__).resolve().parent.parent

    columns = [
        "vendor_name", "model_name", "MYCT", "MMIN", "MMAX",
        "CACH", "CHMIN", "CHMAX", "PRP", "ERP"
    ]

    df = pd.read_csv(
        project_dir / "data" / "machine.data",
        header=None,
        names=columns
    )
    df = df.reset_index(drop=True)

    # PDF-mandated: vendor name and model name are not useful for
    # regression -- discard both.
    #
    # ERP is the ORIGINAL AUTHORS' OWN linear-regression estimate of
    # PRP -- using it as a feature would be leakage. Saved separately,
    # never used as X.
    erp = df["ERP"].to_numpy(dtype=float)

    X = df.drop(columns=["vendor_name", "model_name", "PRP", "ERP"])

    # log_transform=True:  y = ln(PRP) -- justified by the severe right
    #   skew in PRP's histogram (SD 160.8 > mean 105.6 per .names file).
    # log_transform=False: y = PRP directly -- justified by the .names
    #   file's own "Past Usage" note: Kibler & Aha's (1988) INSTANCE-
    #   BASED prediction (the same family as KNN) got "similar results;
    #   no transformations required" vs. the original linear regression.
    #   Worth testing directly rather than assuming either is right.
    if log_transform:
        y = np.log(df["PRP"].to_numpy(dtype=float))
        target_label = "log-PRP"
        output_subdir = "hardware_log"
    else:
        y = df["PRP"].to_numpy(dtype=float)
        target_label = "raw PRP"
        output_subdir = "hardware_raw"

    numerical_cols = ["MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX"]

    results, summary, tuning, predictions = evl.run_regression_5x2(
        X,
        y,
        numerical_cols=numerical_cols,
        categorical_levels=None,
        k_candidates=range(1, 26),
        gamma_candidates=np.logspace(-3, 2, 25),
        epsilon_candidates=(
            np.linspace(0, 2, 11) if log_transform else np.linspace(0, 100, 11)
        ),
        random_state=42
    )

    output_dir = project_dir / "results" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in {
        "fold_scores": results,
        "summary": summary,
        "tuning": tuning,
        "predictions": predictions
    }.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    pd.DataFrame({
        "row_position": np.arange(len(erp)),
        "ERP": erp,
        "PRP": df["PRP"].to_numpy(dtype=float),
        "log_PRP": np.log(df["PRP"].to_numpy(dtype=float))
    }).to_csv(output_dir / "erp_reference.csv", index=False)

    print(f"\nFINAL SUMMARY ({target_label} scale)")
    print(summary.round(4).to_string(index=False))

    return results, summary, tuning, predictions, erp


def compare_hardware_against_erp(log_transform=True, model_names=None):
    """
    Compare your regressors' predictions against ERP, on whichever
    output directory matches log_transform (see run_hardware_experiment).

    When log_transform=True, predictions are stored in log-PRP space
    and are inverse-transformed (np.exp) back to raw PRP here, purely
    for this comparison -- never used for tuning/model selection.
    When log_transform=False, predictions are already in raw PRP space
    and are compared as-is.

    FAIRNESS NOTE: your models' metrics are averaged over every
    out-of-sample test-fold prediction across the full 5x2cv (each row
    contributes up to 5 predictions). ERP is a single fixed value per
    row from the original 1987 in-sample study, with no cross-
    validation structure of its own. Not an identical experimental
    design -- state this when reporting.
    """
    project_dir = Path(__file__).resolve().parent.parent
    output_subdir = "hardware_log" if log_transform else "hardware_raw"
    output_dir = project_dir / "results" / output_subdir

    # keep_default_na=False: pandas treats the literal string "null" as
    # a missing value by default, which would silently drop your null
    # model's rows on read-back otherwise.
    predictions = pd.read_csv(output_dir / "predictions.csv", keep_default_na=False)
    erp_reference = pd.read_csv(output_dir / "erp_reference.csv")

    if model_names is None:
        model_names = predictions["model"].unique().tolist()

    rows = []

    erp_true = erp_reference["PRP"].to_numpy(dtype=float)
    erp_est = erp_reference["ERP"].to_numpy(dtype=float)
    rows.append({
        "model": "ERP (original authors, 1987)",
        "raw_prp_mse": evl.mean_squared_error(erp_true, erp_est),
        "raw_prp_rmse": evl.root_mean_squared_error(erp_true, erp_est),
        "raw_prp_mae": evl.mean_absolute_error(erp_true, erp_est),
        "n_predictions": len(erp_reference)
    })

    for name in model_names:
        subset = predictions[predictions["model"] == name]

        actual = subset["actual"].to_numpy(dtype=float)
        prediction = subset["prediction"].to_numpy(dtype=float)

        if log_transform:
            raw_actual = np.exp(actual)
            raw_prediction = np.exp(prediction)
        else:
            raw_actual = actual
            raw_prediction = prediction

        rows.append({
            "model": name,
            "raw_prp_mse": evl.mean_squared_error(raw_actual, raw_prediction),
            "raw_prp_rmse": evl.root_mean_squared_error(raw_actual, raw_prediction),
            "raw_prp_mae": evl.mean_absolute_error(raw_actual, raw_prediction),
            "n_predictions": len(subset)
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "erp_comparison.csv", index=False)

    print(f"\nERP COMPARISON (raw PRP scale, source: {output_subdir}) --")
    print("for interpretability only, NOT used for tuning/model selection")
    print(comparison.round(3).to_string(index=False))

    return comparison

def run_forestfires_experiment():
    project_dir = Path(__file__).resolve().parent.parent

    columns = [
        "X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI",
        "temp", "RH", "wind", "rain", "area"
    ]

    df = pd.read_csv(
        project_dir / "data" / "forestfires.csv",
        header=0,
        names=columns
    )
    df = df.reset_index(drop=True)

    X = df.drop(columns=["area"])


    y = np.log(df["area"].to_numpy(dtype=float) + 1.0)

    numerical_cols = [
        "X", "Y", "FFMC", "DMC", "DC", "ISI",
        "temp", "RH", "wind", "rain"
    ]

    results, summary, tuning, predictions = evl.run_regression_5x2(
        X,
        y,
        numerical_cols=numerical_cols,
        categorical_levels=None,
        k_candidates=range(1, 26),
        gamma_candidates=np.logspace(-3, 2, 25),
        epsilon_candidates=np.linspace(0, 2, 11),
        random_state=42,
        cyclic_cols=["month", "day"]
    )

    output_dir = project_dir / "results" / "forestfires"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in {
        "fold_scores": results,
        "summary": summary,
        "tuning": tuning,
        "predictions": predictions
    }.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    print("\nFINAL SUMMARY (log-area scale)")
    print(summary.round(4).to_string(index=False))

    return results, summary, tuning, predictions

def run_abalone_experiment():
    project_dir = Path(__file__).resolve().parent.parent

    columns = [
        "Sex", "Length", "Diameter", "Height",
        "Whole weight", "Shucked Weight",
        "Viscera weight", "Shell weight", "Rings"
    ]

    df = pd.read_csv(
        project_dir / "data" / "abalone.data",
        header=None,
        names=columns
    )

    # The Height correction and outlier exclusion are assumptions.
    df = df.loc[
        df["Height"].ne(0) & df["Height"].ne(0.515)
    ].copy()

    df.loc[df["Height"].eq(1.13), "Height"] = 0.113
    df = df.reset_index(drop=True)

    X = df.drop(columns=["Rings"])
    y = df["Rings"].to_numpy(dtype=float)

    results, summary, tuning, predictions = evl.run_regression_5x2(
        X,
        y,
        numerical_cols=columns[1:-1],
        categorical_levels={"Sex": ["F", "I", "M"]},
        k_candidates=range(1, 26),
        gamma_candidates=np.logspace(-3, 2, 25),
        epsilon_candidates=[0, 1, 2, 3, 4, 5, 6, 8, 10],
        random_state=42
    )

    output_dir = project_dir / "results" / "abalone"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, table in {
        "fold_scores": results,
        "summary": summary,
        "tuning": tuning,
        "predictions": predictions
    }.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    print("\nFINAL SUMMARY")
    print(summary.round(4).to_string(index=False))

    return results, summary, tuning, predictions


def test_null_classifier(dataset_name):
    X, y, normalize = du.load_classification_dataset(dataset_name)
    model = cls.NullClassifier()
    errors = evl.cross_validation(model, X, y, evl.classification_error, seed=42, normalize=normalize)
    return errors


def test_knn_classifier(dataset_name):
    X, y, normalize = du.load_classification_dataset(dataset_name)
    errors, params = evl.tuned_cross_validation(cls.KNNClassifier, X, y, evl.classification_error, seed=42, normalize=normalize)
    return errors, params

def test_cnn_classifier(dataset_name):
    X, y, normalize = du.load_classification_dataset(dataset_name)
    errors, params = evl.tuned_cross_validation(cls.CondensedKNNClassifier, X, y, evl.classification_error, seed=42, normalize=normalize)
    return errors, params

def most_common_params(params):
    pairs = params[:, :2].astype(int)

    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    best_index = np.argmax(counts)

    return unique_pairs[best_index]

def run_classification_summary():
    rows = []

    for dataset_name, display_name in [
        ("breast", "Breast Cancer"),
        ("car", "Car"),
        ("vote", "Congressional Vote")
    ]:
        null_errors = test_null_classifier(dataset_name)
        rows.append([display_name, "Null", "-", "-", np.mean(null_errors)])

        knn_errors, knn_params = test_knn_classifier(dataset_name)
        knn_k, knn_p = most_common_params(knn_params)
        rows.append([display_name, "KNN", knn_k, knn_p, np.mean(knn_errors)])

        cnn_errors, cnn_params = test_cnn_classifier(dataset_name)
        cnn_k, cnn_p = most_common_params(cnn_params)
        rows.append([display_name, "CNN", cnn_k, cnn_p, np.mean(cnn_errors)])

    return rows


def print_classification_summary(rows):
    lines = []

    lines.append(f"{'Dataset':<22} {'Model':<8} {'k':<5} {'p':<5} {'Classification Error':<12}")
    lines.append("-" * 58)

    for dataset, model, k, p, error in rows:
        lines.append(f"{dataset:<22} {model:<8} {str(k):<5} {str(p):<5} {error:.4f}")

    output = "\n".join(lines)

    print(output)

    return output

def run_classification_experiment():
    results = run_classification_summary()
    output = print_classification_summary(results)

    output_dir = Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "classification_summary.txt", "w") as file:
        file.write(output)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True,
        choices=["test-classification", "abalone", "forestfires", "hardware"]
    )

    args = parser.parse_args()

    if args.task == "test-classification":
        run_classification_experiment()

    elif args.task == "abalone":
        run_abalone_experiment()

    elif args.task == "forestfires":
        run_forestfires_experiment()

    elif args.task == "hardware":
        run_hardware_experiment(log_transform=True)
        run_hardware_experiment(log_transform=False)
        compare_hardware_against_erp()


