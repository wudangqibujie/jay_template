from typing import List


class Solution:
    def matrixSum(self, nums: List[List[int]]) -> int:
        for ix, l in enumerate(nums):
            l.sort(reverse=True)
        vl = 0
        for j in range(len(nums[0])):
            vl += max([nums[i][j] for i in range(len(nums))])
        return vl

s = Solution()
nums = [[7,2,1],[6,4,2],[6,5,3],[3,2,1]]
nums = [[1]]
print(s.matrixSum(nums))