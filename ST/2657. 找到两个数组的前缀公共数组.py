from typing import List


class Solution:
    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        rslt = []
        for ix in range(1, len(A)):
            rslt.append(len(set(A[:ix]) & set(B[:ix])))
        rslt.append(len(A))
        return rslt


A = [1,3,2,4]
B = [3,1,2,4]
s = Solution()
A = [2,3,1]
B = [3,1,2]
print(s.findThePrefixCommonArray(A, B))