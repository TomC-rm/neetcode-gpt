import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))
        
        delta = (y_hat - y_true) * y_hat * (1 - y_hat)
        dL_dw = delta * x
        dL_db = delta

        return (np.round(dL_dw, 5), np.round(dL_db, 5))
