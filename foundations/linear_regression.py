import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        Y_chap=X*weights
        Y_chap=np.sum(Y_chap, axis=1) #on met axis 1 comme ça il calcule pour chaque sous liste et pas total complet
        #pour info: on avait aussi np.dot(X,weights) qui faisait pareil comme c'est un dot product
        return np.round(Y_chap,5)
        

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places

        MSE=np.mean((model_prediction-ground_truth)**2)
        return np.round(MSE,5)
