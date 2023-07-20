from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [nums[0]]
        for ix in range(1, len(nums)):
            dp.append(max(nums[ix], dp[-1] + nums[ix]))
        return max(dp)


s = Solution()
nums = [5]
print(s.maxSubArray(nums))
