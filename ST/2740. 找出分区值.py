from typing import List

class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        val = float('inf')
        for ix in range(1, len(nums)):
            val = min(val, nums[ix] - nums[ix - 1])
        return val


s = Solution()
nums = [100,1,10]
print(s.findValueOfPartition(nums))