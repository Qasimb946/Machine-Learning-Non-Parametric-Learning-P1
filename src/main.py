"""
This file is the driver for project 1
"""
import numpy as np
import data_utils as du
import classification as cls
import evaluation as evl

import pandas as pd
from pathlib import Path

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

    # PDF-mandated log transform on the target only -- never z-score
    # area, never inverse-transform before computing loss.
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

    lines.append(f"{'Dataset':<22} {'Model':<8} {'k':<5} {'p':<5} {'Mean Error':<12}")
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
        choices=["test-classification", "abalone", "forestfires"]
    )

    args = parser.parse_args()

    if args.task == "test-classification":
        run_classification_experiment()

    elif args.task == "abalone":
        run_abalone_experiment()

    elif args.task == "forestfires":
        run_forestfires_experiment()


