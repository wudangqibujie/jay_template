from typing import List


class Solution:
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        dp = [[0], [0]]
        for ix in range(len(arr)):
            tmp_dp = []
            st = ix
            step = 1
            while st >= 0 and st > ix - k:
                val = max(dp[-step]) + step * max(arr[st:ix+1])
                step += 1
                st -= 1
                tmp_dp.append(val)
            dp.append(tmp_dp)
        return max(dp[-1])


s = Solution()
arr = [1,15,7,9,2,5,10]
k = 3
arr = [1,4,1,5,7,3,6,1,9,9,3]
k = 4
arr = [1]
k = 1
print(s.maxSumAfterPartitioning(arr, k))