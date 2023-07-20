from typing import List

class Solution:
    def findNonMinOrMax(self, nums: List[int]) -> int:
        not_val = [max(nums), min(nums)]

        for i in nums:
            if i not in not_val:
                return i
        return -1