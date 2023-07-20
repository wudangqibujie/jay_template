from typing import List


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1]
        for ix in range(1, len(nums)):
            jx = ix - 1
            rs = 0
            while jx >= 0:
                if nums[ix] > nums[jx]:
                    rs = max(rs, dp[jx])
                jx -= 1
            dp.append(rs + 1)
        return max(dp)

s = Solution()
print(s.lengthOfLIS([10]))

# ps -ef|grep aaa|grep -v grep|awk ' {print \"kill -9 \" $2}' |sh