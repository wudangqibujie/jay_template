from typing import List



class Solution:
    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        for ix, i in enumerate(nums):
            if i < pivot:
                continue
            j = ix + 1
            while j < len(nums) and nums[j] >= pivot:
                j += 1
            if j == len(nums):
                continue
            nums[ix], nums[j] = nums[j], nums[ix]
        print(nums)
        for ix, i in enumerate(nums):
            if i <= pivot:
                continue
            j = ix + 1
            while j < len(nums) and nums[j] > pivot:
                j += 1
            if j == len(nums):
                continue
            nums[ix], nums[j] = nums[j], nums[ix]
        print(nums)


s = Solution()
nums = [9,12,5,10,14,3,10]
pivot = 10
nums = [-3,4,3,2]
pivot = 2
s.pivotArray(nums, pivot)