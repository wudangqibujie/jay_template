from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if nums[0] == 0:
            return False
        dp = [1]
        for ix in range(1, len(nums)):
            if dp[-1] == 0:
                dp.append(0)
            else:
                if nums[ix] == 0:
                    dp.append(0)
                else:
                    dp.append(1)
        return not dp[-1] == 0


s = Solution()
nums = [2,3,1,1,4]
nums = [3, 2, 1, 0, 4]
print(s.canJump(nums))