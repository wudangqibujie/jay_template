from typing import List


class Solution:
    def minPairSum(self, nums: List[int]) -> int:
        nums.sort()
        rs = []
        ix = 0
        # print(nums)
        while ix < len(nums) // 2:
            rs.append(nums[ix] + nums[-(ix + 1)])
            ix += 1
        # print(rs)
        return max(rs)

s = Solution()
nums = [3,5,4,2,4,6]
nums = [3,5]
print(s.minPairSum(nums))