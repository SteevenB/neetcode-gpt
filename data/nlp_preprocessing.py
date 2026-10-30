import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)

        #Would be good to simplify all of this
        ans=set()

        every_sentences=[]
        for phrases in positive+negative:
            words=set(phrases.split())
            ans.update(words)

        ans=sorted(ans)
        
        ans = {mot: index for index, mot in enumerate(ans, start=1)} #go through to put word:number, instead of number:word

        for phrases in positive+negative:
            temp_phrase=[]
            words=phrases.split()
            for i in words:
                num=ans[i]
                temp_phrase.append(num)
            every_sentences.append(torch.tensor(temp_phrase))
        
        every_sentences=nn.utils.rnn.pad_sequence(every_sentences, batch_first=True)

        return every_sentences


        
