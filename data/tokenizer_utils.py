from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        results=[]

        for num in numbers: #we go through each number in the list
            strnum=str(num) #convert to char
            tokens=[]

            start=0
            n=len(strnum) #we determine the "size" of the number for our loops

            while(start<n):
                match=False

                for end in range(n,start,-1): #we start with the biggest size possible until we reach the smallest

                    part=strnum[start:end] #slice through loops to get our new char number to check if it's in vocab

                    if part in vocab:
                        tokens.append(part) #we add our part in tokens for this number
                        start=end #we change the start to not start from the same pos, we start later in the number
                        match=True
                        break #go out of the for loop, to go back in the while loop, we will come back if the new start is not the last position of the number

                if not match: #if there is never a part in vocab, we still need to go through so we just do start+1 here until it's finished, ideally raise an error
                    tokens.append(strnum[start])
                    start+=1 

            results.append(tokens)

        return results


    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        #almost the same, no int to str conversion, we just work with text, also no saves of tokens as we are just counting
        counting=0

        start=0
        n=len(text) 

        while(start<n):
            match=False

            for end in range(n,start,-1): 

                part=text[start:end] 

                if part in vocab:
                    counting+=1
                    start=end 
                    match=True
                    break 

            if not match:
                start+=1 

        return counting

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.

        count=self.count_tokens(text, vocab)
        separated_text=text.split()

        if not separated_text: #if no words, generally zero
            return 0.0

        
        result=count/len(separated_text)
        return round(result,4)
