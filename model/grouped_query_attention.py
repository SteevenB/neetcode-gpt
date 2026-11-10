import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        q=q.reshape(B, T, self.num_heads, self.head_dim)
        k=k.reshape(B, T, self.num_kv_heads, self.head_dim)
        v=v.reshape(B, T, self.num_kv_heads, self.head_dim)

        groups = self.num_heads // self.num_kv_heads

        k=k.repeat_interleave(groups,dim=2)
        v=v.repeat_interleave(groups, dim=2) #dim=2 is num_kv_heads

        scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) #should have used permute
        scores = scores / (q.shape[-1] ** 0.5)

        causal_mask=torch.tril(torch.ones(x.shape[1],x.shape[1])) #we need context shape by context shape matrix
        masked_scores=scores.masked_fill(causal_mask==0,float('-inf'))

        scores=torch.softmax(masked_scores, dim=-1)

        result= scores @ v.transpose(1, 2) #transpose V to respect shape

        result = result.transpose(1, 2) #get back to the correct order
        result = result.reshape(B, T, self.num_heads * self.head_dim)

        result = self.output_proj(result)

        return torch.round(result, decimals=4)




