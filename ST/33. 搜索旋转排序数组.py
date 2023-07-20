from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 1:
            rslt = 0 if nums[0] == target else -1
            return rslt

        def bin_find(ix, jx, target):
            i, j = ix, jx
            while i < j:
                mid = (i + j) // 2
                if nums[mid] > target:
                    j = mid
                elif nums[mid] < target:
                    i = mid + 1
                else:
                    return mid
            return -1

        # print(bin_find(nums, target))

        i, j = 0, len(nums) - 1
        while i < j:
            mid = (i + j) // 2
            if nums[i] <= nums[mid] <= nums[j]:
                return bin_find(i, j + 1, target)
            if nums[mid] > nums[i] and nums[mid] > nums[j]:
                i = mid
            else:
                j = mid

        if nums[-1] == target:
            return len(nums) - 1
        elif nums[-1] > target:
            ix, jx = i + 1, len(nums)
        else:
            ix, jx = 0, j + 1

        return bin_find(ix, jx, target)

s = Solution()
nums = [4,5,6,7,0,1,2]
target = 3
print(s.search(nums, target))