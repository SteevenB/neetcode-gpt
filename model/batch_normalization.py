import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        #without torch.tensors here, we convert to numpy array to make numpy multiplication of matrices
        running_mean = np.array(running_mean)
        running_var=np.array(running_var)

        if(training):
            #axis=0 for batch, calculates per column instead of going one by one
            mean=np.mean(x,axis=0)
            var=np.var(x,axis=0)
            x_hat=(x-mean)/(np.sqrt(var+eps))
            y=gamma*x_hat+beta

            running_mean=(1-momentum)*running_mean+momentum*mean
            running_var=(1-momentum)*running_var+momentum*var
        
        else:
            x_hat=(x-running_mean)/(np.sqrt(running_var+eps))
            y=gamma*x_hat+beta

        return(np.round(y.tolist(),4),np.round(running_mean.tolist(),4),np.round(running_var.tolist(),4))
