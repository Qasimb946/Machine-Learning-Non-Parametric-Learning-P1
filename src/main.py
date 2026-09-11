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


def test_KNN_classifier():
    breast_cancer = du.load_data("data/breast-cancer-wisconsin.data", missing_value="?")
    X = breast_cancer[:, 1:-1].astype(float)
    y = breast_cancer[:, -1]



if __name__ == '__main__':
    null_classifier_error = test_null_classifier()
    print(null_classifier_error)
    pass
