from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)

        unique_chars = sorted(list(set(text))) #set removes duplicates, list to get char, sorted to do it alphabetically

        stoi = {car: i for i, car in enumerate(unique_chars)}
        itos = {i: car for i, car in enumerate(unique_chars)}

        return (stoi, itos)


        

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        return [stoi[i] for i in text]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        return "".join(itos[i] for i in ids) #will connect the list of characters into a string, join with "" so nothing, example "-" it would have been h-e-l-l-o
