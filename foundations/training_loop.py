import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        
        w = np.zeros(X.shape[1])
        b=0

        print("X :", X.shape)
        print("w :", w.shape)
        print("b :", np.shape(b))

        for _ in range (epochs):
            y_hat=X @ w +b
            mse=np.mean((y_hat-y)**2)

            der_L_w=2*np.transpose(X) @ (y_hat-y)/len(X) #USE @ NOT * 
            der_L_b=2*np.mean(y_hat-y)

            w-=lr*der_L_w
            b-=lr*der_L_b

        return (np.round(w,5),round(b,5))
