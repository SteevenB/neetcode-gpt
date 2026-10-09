import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)

        y_pred=np.clip(y_pred, 1e-7, 1-1e-7)

        #au passege, on fait la moyenne comme c'est somme des éléments divisés par le nbr total d'éléments présents
        l=(-1/len(y_true))*(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

        return round(np.sum(l),4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        y_pred=np.clip(y_pred,1e-7,1-1e-7)

        total=(-1/len(y_true))*y_true*np.log(y_pred)

        return round(np.sum(total), 4)