import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x_arr = np.array(x, dtype=np.float64)
        w1_arr = np.array(W1, dtype=np.float64)
        w2_arr = np.array(W2, dtype=np.float64)
        b1_arr = np.array(b1, dtype=np.float64)
        b2_arr = np.array(b2, dtype=np.float64)
        y_true_arr = np.array(y_true, dtype=np.float64)

        # ==================
        # Forward Pass
        # ==================

        # Linear layer 1
        z1 = np.matmul(w1_arr, x_arr) + b1_arr

        # Activation ReLU
        a1 = np.maximum(0, z1)

        # Linear layer 2
        z2 = np.matmul(w2_arr, a1) + b2_arr

        y_hat = z2
        loss = np.mean((y_hat - y_true_arr) ** 2)
        n = len(y_true_arr)

        # ==================
        # Backward Pass
        # ==================

        # Output gradient
        dz2 = 2/n * (y_hat - y_true_arr)

        # Layer 2 gradients
        dw2 = np.outer(dz2, a1)
        db2 = dz2

        # Error backpropagated through W2 to hidden activation a1
        da1 = np.matmul(w2_arr.T, dz2)

        # Error backpropagated through ReLU mask
        dz1 = da1 * (z1 > 0).astype(np.float64)

        # Layer 1 Gradients
        dw1 = np.outer(dz1, x_arr)
        db1 = dz1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dw1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dw2, 4).tolist(),
            'db2': np.round(db2, 4).tolist()
        }













