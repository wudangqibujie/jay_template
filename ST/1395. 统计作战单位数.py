from typing import List

class Solution:
    def numTeams(self, rating: List[int]) -> int:

        cnt = 0


        def search(nums):
            nonlocal cnt
            dp = [0 for _ in range(len(nums))]
            dp[1] = int(nums[1] > nums[0])
            for ix in range(2, len(nums)):
                jx = ix - 1
                tmp_cnt = 0
                while jx >= 0:
                    if nums[ix] > nums[jx]:
                        tmp_cnt += 1
                    jx -= 1
                dp[ix] = tmp_cnt
            for ix in range(2, len(nums)):
                jx = ix - 1
                while jx >= 1:
                    if dp[jx] == 0:
                        jx -= 1
                        continue
                    if nums[ix] > nums[jx]:
                        cnt += dp[jx]
                    jx -= 1
            dp = [0 for _ in range(len(nums))]
            dp[1] = int(nums[1] < nums[0])
            for ix in range(2, len(nums)):
                jx = ix - 1
                tmp_cnt = 0
                while jx >= 0:
                    if nums[ix] < nums[jx]:
                        tmp_cnt += 1
                    jx -= 1
                dp[ix] = tmp_cnt
            for ix in range(2, len(nums)):
                jx = ix - 1
                while jx >= 1:
                    if dp[jx] == 0:
                        jx -= 1
                        continue
                    if nums[ix] < nums[jx]:
                        cnt += dp[jx]
                    jx -= 1

        search(rating)
        return cnt


s = Solution()
print(s.numTeams([2, 5, 3, 4, 1, 2, 5, 3, 4, 1, 2, 5, 3, 4, 1, 2, 5, 3, 4, 1, 2, 5, 3, 4, 1, 2, 5, 3, 4, 1, 2, 5, 3, 4, 1, 2, 5, 3, 4, 1]))