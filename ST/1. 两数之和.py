from typing import List


class Solution:
    def ABC(self, nums: List[int], target: int) -> List[int]:
        log = dict()
        for ix, n in enumerate(nums):
            log[n] = ix
        for ix, n in enumerate(nums):
            if target - n in log and log[target - n] != ix:
                return [ix, log[target - n]]
