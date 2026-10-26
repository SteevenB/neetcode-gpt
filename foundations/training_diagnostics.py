import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        with torch.no_grad():

            ans=[]
            for layer in model.children(): #go through each layer of the model 
                x=layer(x)
                if isinstance(layer,nn.Linear): #.item to not get tensor everywhere and only get the float value
                    mean=torch.round(torch.mean(x),decimals=4).item()
                    std=torch.round(torch.std(x),decimals=4).item()

                    dead_fraction = torch.round(torch.mean((x <= 0).all(dim=0).float()), decimals=4).item() #to calculate part of dead neurons inside the batch (dim=0)
                    stats = {
                        "mean": mean,
                        "std": std,
                        "dead_fraction": dead_fraction
                    }
                    ans.append(stats)

        return ans

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.

        model.zero_grad()
        ans=[]
        criterion=nn.MSELoss()

        for layer in model.children():
            x=layer(x)

        loss=criterion(x,y)
        loss.backward() #call first MSELoss (to be able to use it) in criterion, we go backward and re loop with the bckwrd propagation to do our stats

        for layer in model.children():
            if isinstance(layer,nn.Linear):
                mean=torch.round(torch.mean(layer.weight.grad),decimals=4).item()
                std=torch.round(torch.std(layer.weight.grad),decimals=4).item()
                norm=torch.round(torch.norm(layer.weight.grad),decimals=4).item() #deprecated, use torch.linalg.vector_norm or blablabla.matrix_norm

                stats={
                    "mean":mean,
                    "std":std,
                    "norm":norm
                }
                ans.append(stats)

        return ans


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)

        # Dead Neurons
        if any(layer["dead_fraction"] > 0.5 for layer in activation_stats): #we check if any layer's dead fration is >0.5, etc
            return "dead_neurons"
            
        # Exploding gradients
        if any(layer["norm"] > 1000 for layer in gradient_stats) or any(layer["std"] > 10.0 for layer in activation_stats):
            return "exploding_gradients"
            
        # Vanishing gradients
        if any(layer["norm"] < 1e-5 for layer in gradient_stats) or any(layer["std"] < 0.1 for layer in activation_stats):
            return "vanishing_gradients"


        return "healthy"
