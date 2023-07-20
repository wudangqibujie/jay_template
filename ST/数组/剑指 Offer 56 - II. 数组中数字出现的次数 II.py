from typing import List

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        d = dict()
        for n in nums:
            if n in d:
                d[n] += 1
            else:
                d[n] = 1
        for k, v in d.items():
            if v == 1:
                return k
        return -1
