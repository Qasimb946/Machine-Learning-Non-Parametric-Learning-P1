"""
This file includes all classification models
"""
import numpy as np


class NullClassifier:
    def __init__(self):
        self.majority_class = None

    def fit(self, X_train, y_train):
        counts = {}
        for value in y_train:
            if value in counts:
                counts[value] += 1
            else:
                counts[value] = 1

        self.majority_class = max(counts, key=counts.get)

    def predict(self, x):
        return np.full(len(x), self.majority_class)


class KNNClassifier:
    def __init__(self, k, p):
        self.k = k
        self.p = p
        self.X_train = None
        self.y_train = None


    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        prediction = []
        for x in X_test:
            distances = np.sum(np.abs(self.X_train - x)**self.p, axis=1)**(1/self.p)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_nearest_indices]
            labels, counts = np.unique(k_nearest_labels, return_counts=True)
            max_count = np.max(counts)
            tied_classes = labels[counts == max_count]
            predicted_label = np.random.choice(tied_classes)
            prediction.append(predicted_label)
        return np.array(prediction)

class CondensedKNNClassifier:
    pass
