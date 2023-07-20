from typing import List


class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        ix, jx = 0, len(nums) - 1
        while ix < jx:
            while ix < jx:
                if nums[ix] == 2:
                    break
                ix += 1
            while ix < jx:
                if nums[jx] == 0:
                    break
                jx -= 1
            nums[ix], nums[jx] = nums[jx], nums[ix]
            ix += 1
            jx -= 1

        ix, jx = 0, len(nums) - 1
        while ix < jx:
            while ix < jx:
                if nums[ix] == 1:
                    break
                ix += 1
            while ix < jx:
                if nums[jx] == 0:
                    break
                jx -= 1
            nums[ix], nums[jx] = nums[jx], nums[ix]
            ix += 1
            jx -= 1

        ix, jx = 0, len(nums) - 1
        while ix < jx:
            while ix < jx:
                if nums[ix] == 2:
                    break
                ix += 1
            while ix < jx:
                if nums[jx] == 1:
                    break
                jx -= 1
            nums[ix], nums[jx] = nums[jx], nums[ix]
            ix += 1
            jx -= 1

s = Solution()
nums = [2, 0, 2, 1, 1, 0]
nums = [2, 0, 1]
s.sortColors(nums)
print(nums)