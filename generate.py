import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()

        generated_text=""
        for i in range(new_chars):
            context=context[:,-context_length:]
            logits=model(context)
            last_pos=logits[:,-1,:]
            res=torch.softmax(last_pos,dim=-1)

            next_token=torch.multinomial(res,1,generator=generator) #1 is number of samples, so 1 word

            # YOUR CODE (arbitrary number of lines)
            # The line where you call torch.multinomial(). Pass in the generator as well.
            generator.set_state(initial_state)
            # MORE OF YOUR CODE (arbitrary number of lines)

            context = torch.cat((context, next_token), dim=1)

            generated_text+=int_to_char[next_token.item()] #reminder: next_token is a tensor

        return generated_text
