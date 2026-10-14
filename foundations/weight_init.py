import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list

        torch.manual_seed(0)

        std=math.sqrt(2/(fan_in+fan_out))
        w=torch.randn(fan_out,fan_in)*std
        
        return w.round(decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list

        torch.manual_seed(0)

        std=math.sqrt(2/fan_in)
        w=torch.randn(fan_out,fan_in)*std
        
        return w.round(decimals=4).tolist()
        

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        torch.manual_seed(0)
        # Use torch.manual_seed(0) once at the start.

        current_dim=input_dim
        res=[]
        weights=[]

        # Generate all weights before generating the input
        for _ in range(num_layers):

            if init_type=="xavier": #problem I encountered, trying to call functions before use random seeding again so results are not what we want, instead implement already inside here
                std=math.sqrt(2/(current_dim + hidden_dim))

            elif init_type=="kaiming":
                std=math.sqrt(2/current_dim)
                
            else:
                std=1.0 #if random
            
            w=torch.randn(hidden_dim, current_dim)*std
            weights.append(w)

            current_dim = hidden_dim
        #our x to check
        x = torch.randn(1, input_dim)

        # We then go through each weight we have with our only input x (error I did first, only one loop with x inside which change, here it stays the same with 2 loops, and weights separated)
        for w in weights:
            
            # x @ w.T
            x = x @ w.T
            
            #  Relu if Kaiming, not here because it's said we do relu for every case
            #if init_type == "kaiming":
            x = torch.relu(x)

            res.append(round(x.std().item(), 2)) # bad choice from me, naming a variable std sorry :(

        return res

        # Return the std of activations after each layer, rounded to 2 decimals.