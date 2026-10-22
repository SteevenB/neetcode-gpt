import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        rms=np.sqrt(np.mean(np.square(x))+eps) #use np.square because list and not int (so no x**2)
        x_hat=x/rms
        
        output=gamma*x_hat
        # Return result rounded to 4 decimal places as a list
        return np.round(output,4)
