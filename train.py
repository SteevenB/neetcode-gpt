import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.

        
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            torch.manual_seed(epoch)
            rand_start=torch.randint(0,len(data)-context_length,size=(batch_size,))
            X=[]
            Y=[]

            for i in range(batch_size): #get our batch of data to train after
                sample_X=data[rand_start[i]:rand_start[i]+context_length]
                sample_Y=data[rand_start[i]+1:rand_start[i]+1+context_length]
                X.append(sample_X)
                Y.append(sample_Y)

            X=torch.stack(X) #stack so they're following eachother
            Y=torch.stack(Y)

            logits=model(X) #train

            logits=logits.reshape(logits.shape[0]*logits.shape[1],logits.shape[2]) #we need for crossentropy to flatten to 2 dim (before:batch_size,context_length,vocab_size)
            Y=Y.reshape(Y.shape[0]*Y.shape[1],)

            loss=F.cross_entropy(logits,Y) #F is pytorch.functional (see above), check loss between Y and logits
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(),4)

            




