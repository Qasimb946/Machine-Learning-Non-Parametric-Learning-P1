"""
This file includes all classification models
"""
import numpy as np


class NullClassifier:
    def __init__(self):
        self.majority_class = None

    def fit(self, X, y):
        counts = {}
        for value in y:
            if value in counts:
                counts[value] += 1
            else:
                counts[value] = 1

        self.majority_class = max(counts, key=counts.get)

    def predict(self, x):
        return np.full(len(x), self.majority_class)


class KNNClassifier:
    pass

class CondensedKNNClassifier:
    pass
