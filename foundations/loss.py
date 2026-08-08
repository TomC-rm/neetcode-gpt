import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        loss = -np.mean(y_true * np.log(y_pred + 1e-7) + (1-y_true) * np.log(1 - y_pred + 1e-7))
        return round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # return round(your_answer, 4)
        loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-7), axis=1))
        return round(loss, 4)
