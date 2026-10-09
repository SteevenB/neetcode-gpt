import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float],
    ) -> dict:

        #convert inputs to numpy arrays to use transpose and @ and size
        x = np.asarray(x, dtype=float)
        W1 = np.asarray(W1, dtype=float)
        b1 = np.asarray(b1, dtype=float)
        W2 = np.asarray(W2, dtype=float)
        b2 = np.asarray(b2, dtype=float)
        y_true = np.asarray(y_true, dtype=float)

        #forward pass
        z1 = x @ W1.T + b1

        #ReLU activation
        a1 = np.maximum(0, z1)

        #y_hat
        y_hat = a1 @ W2.T + b2

        #loss
        loss = np.mean((y_hat - y_true) ** 2)

        #Backward
        dz2 = 2 * (y_hat - y_true) / y_true.size

        #gradients for second linear layer
        db2 = dz2.copy()
        dW2 = np.outer(dz2, a1)

        #gradient flowing into hidden activation
        da1 = dz2 @ W2

        #ReLU derivative: 1 only when z1 > 0
        dz1 = da1 * (z1 > 0)

        #gradients for first linear layer
        db1 = dz1.copy()
        dW1 = np.outer(dz1, x)

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }