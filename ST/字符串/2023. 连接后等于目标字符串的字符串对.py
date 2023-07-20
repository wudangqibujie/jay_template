from typing import List


class Solution:
    def numOfPairs(self, nums: List[str], target: str) -> int:
        rslt = 0
        for ix1 in range(len(nums) - 1):
            for ix2 in range(ix1 + 1, len(nums)):
                if nums[ix1] + nums[ix2] == target:
                    rslt += 1
                if nums[ix2] + nums[ix1] == target:
                    rslt += 1
        return rslt


s = Solution()
nums = ["777","7","77","77"]
target = "7777"
nums = ["123","4","12","34"]
target = "1234"
nums = ["1"]
target = "11"
print(s.numOfPairs(nums, target))