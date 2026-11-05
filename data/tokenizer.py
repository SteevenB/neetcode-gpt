from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed

        decomposed=list(corpus)
        history_merges=[]
        for _ in range(num_merges): #number of merges to do
            total=[]
            for i in range(len(decomposed)-1):
                result=(decomposed[i],decomposed[i+1])
                total.append(result) #we register all our pair of letters from start to finish (a,b then b,c then c,a etc)

            sorted_elements=Counter(total).most_common() #counter the most common pairs present
            
            winner=sorted(sorted_elements,key=lambda element:(-element[1],element[0])) #to break tie, (- to get for example -5<-2 instead of 2<5 so it doesn't reverse when returning frequencies|| also, element 1 first because we want to sort by frequency first, and then alphabetically)

            history_merges.append(winner[0][0]) #we register our winning pair for our history for this merge

            #then we merge our winning pair in our decomposed list

            next_decomposed=[]
            j=0

            while (j<len(decomposed)):
                if( j<len(decomposed)-1 and (decomposed[j],decomposed[j+1])==winner[0][0]): 
                    next_decomposed.append(decomposed[j]+decomposed[j+1])
                    j+=2
                else:
                    next_decomposed.append(decomposed[j])
                    j+=1

            decomposed=next_decomposed #we change the corpus to get the new one merged

        return history_merges

        
        
                 

        
