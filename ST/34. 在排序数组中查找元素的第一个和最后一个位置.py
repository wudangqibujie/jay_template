from typing import List

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1, -1]
        lst_i = None
        i, j = 0, len(nums)
        while i < j:
            mid = (i + j) // 2
            if nums[mid] == target:
                lst_i = mid
                i = mid + 1
            elif nums[mid] < target:
                i = mid + 1
            else:
                j = mid
        if lst_i is None:
            return [-1, -1]
        lst_j = None
        i, j = 0, len(nums)
        while i < j:
            mid = (i + j) // 2
            if nums[mid] == target:
                lst_j = mid
                j = mid
            elif nums[mid] < target:
                i = mid + 1
            else:
                j = mid

        return [lst_j, lst_i]

s = Solution()
nums = [1, 1, 2, 2, 2, 2, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 10]
target = 11

print(s.searchRange(nums, target))