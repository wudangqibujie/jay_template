from typing import List


class Solution:
    def rearrangeArray(self, nums: List[int]) -> List[int]:

        rslt = []
        i1, i2 = 0, 0
        while i1 < len(nums) or i2 < len(nums):
            while i1 < len(nums) and nums[i1] < 0:
                i1 += 1
            while i2 < len(nums) and nums[i2] > 0:
                i2 += 1
            if i1 < len(nums):
                rslt.append(nums[i1])
            if i2 < len(nums):
                rslt.append(nums[i2])
            i1 += 1
            i2 += 1
        return rslt


s = Solution()
nums = [3,1,-2,-5,2,-4]
nums = [-1,1]
print(s.rearrangeArray(nums))