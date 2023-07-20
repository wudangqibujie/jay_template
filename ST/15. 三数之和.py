from typing import List

class Solution:
    def ABC(self, nums: List[int]) -> List[List[int]]:
        rslt = []
        nums.sort()
        print(nums)
        for ix in range(1, len(nums) - 1):
            i, j = 0, len(nums) - 1
            while i < ix and ix < j:
                if nums[i] + nums[ix] + nums[j] == 0:
                    if (nums[i], nums[ix], nums[j]) not in rslt:
                        rslt.append((nums[i], nums[ix], nums[j]))
                    i += 1
                    j -= 1
                elif nums[i] + nums[ix] + nums[j] < 0:
                    i += 1
                else:
                    j -= 1
        return rslt


s = Solution()
nums = [-1,0,1,2,-1,-4]
nums = [0, 1, 1]
# nums = [0, 0, 0, 0]
# nums = [3,0,-2,-1,1,2]
print(s.ABC(nums))