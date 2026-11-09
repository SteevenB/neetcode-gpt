import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split() 
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        #this exercice is simplified, no transformation in numbers, we only focus on splitting text (if I understand correctly)
        torch.manual_seed(0)

        splitted=raw_dataset.split()
        start_random=torch.randint(0,len(splitted)-context_length,size=(batch_size,))

        X=[]
        Y=[]

        for i in range(batch_size):
            start=start_random[i].item()
            one_item_X=splitted[start:start+context_length]
            one_item_Y=splitted[start+1:start+1+context_length]

            X.append(one_item_X)
            Y.append(one_item_Y)
        

        return (X,Y)






