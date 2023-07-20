from typing import List


class Solution:
    def triangularSum(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0] % 10
        rslt = []
        for ix in range(1, len(nums)):
            rslt.append((nums[ix] + nums[ix - 1]) % 10)
        while len(rslt) > 1:
            rslt = [rslt[ix] + rslt[ix + 1] for ix in range(len(rslt) - 1)]
        return rslt[0] % 10

s = Solution()
nums = [1,2,3,4,15]
nums = [18, 33]
print(s.triangularSum(nums))



