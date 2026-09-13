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


def test_null_classifier():
    breast_cancer = du.load_data("data/breast-cancer-wisconsin.data", missing_value="?")

    X = breast_cancer[:, 1:-1].astype(float)
    y = breast_cancer[:, -1]

    model = cls.NullClassifier()
    errors = evl.cross_validation(model, X, y, evl.classification_error)
    return errors


def test_knn_classifier():
    results = []

    breast_cancer = du.load_data("data/breast-cancer-wisconsin.data", missing_value="?")
    X = breast_cancer[:, 1:-1].astype(float)
    y = breast_cancer[:, -1]

    for p in [1, 2]:
        for k in range(1, 10):
            model = cls.KNNClassifier(k=k, p=p)
            errors = evl.cross_validation(model, X, y, evl.classification_error, seed=42)
            results.append([k, p, np.mean(errors)])

    results = np.array(results)
    best_index = np.argmin(results[:, 2])
    best_result = results[best_index]
    return best_result


def test_cnn_classifier():
    results = []

    breast_cancer = du.load_data("data/breast-cancer-wisconsin.data", missing_value="?")
    X = breast_cancer[:, 1:-1].astype(float)
    y = breast_cancer[:, -1]

    for p in [1, 2]:
        for k in range(1, 10):
            model = cls.CondensedKNNClassifier(k=k, p=p)
            errors = evl.cross_validation(model, X, y, evl.classification_error, seed=42)
            results.append([k, p, np.mean(errors)])
    results = np.array(results)
    best_index = np.argmin(results[:, 2])
    best_result = results[best_index]

    return best_result



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        required=True,
        choices=["null-classifier", "knn-classifier", "abalone", "forestfires"]
    )

    args = parser.parse_args()

    if args.task == "null-classifier":
        print(test_null_classifier())

    elif args.task == "knn-classifier":
        print(test_knn_classifier())

    elif args.task == "abalone":
        run_abalone_experiment()

    elif args.task == "forestfires":
        run_forestfires_experiment()
