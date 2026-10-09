import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp

        x=z-np.max(z)

        x=np.exp(x)

            # return np.round(your_answer, 4)

        tot=sum(x)

        x=x/tot
        return np.round(x,4)
