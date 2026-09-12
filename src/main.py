"""
This file is the driver for project 1
"""
import numpy as np
import data_utils as du
import classification as cls
import evaluation as evl


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

if __name__ == '__main__':
    # null_classifier_errors = test_null_classifier()
    # print(null_classifier_errors)

    knn_classifier_result = test_knn_classifier()
    print(knn_classifier_result)
