import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value

        #So we have three linear layers, each with embedding (input dimension) and each with attention_dim (output dimension)
        if embedding_dim>0:
            self.key=nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query=nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value=nn.Linear(embedding_dim, attention_dim, bias=False)
            self.softmax=nn.Softmax(dim=2)
        else:
            print("error: embedding_dim must be greater than 0")


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places

        trueKey=self.key(embedded)
        trueQuery=self.query(embedded)
        trueValue=self.value(embedded)

        attention_score=(trueQuery @ trueKey.transpose(1,2))/torch.sqrt(torch.tensor(trueKey.shape[2])) #we remove the batch size with transpose 1,2 for calculations, truekey shape is (..,.., att_dim)
        causal_mask=torch.tril(torch.ones(embedded.shape[1],embedded.shape[1])) #we need context shape by context shape matrix
        masked_scores=attention_score.masked_fill(causal_mask==0,float('-inf'))

        scores=self.softmax(masked_scores)

        result= scores @ trueValue
        
        return torch.round(result, decimals=4)
