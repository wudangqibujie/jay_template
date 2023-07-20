from typing import List


class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        dp = [triangle[0]]

        for ix in range(1, len(triangle)):
            tmp_lst = []
            for jx in range(len(triangle[ix])):
                can_idx = [jx, jx - 1]
                rs = float('inf')
                for cix in can_idx:
                    if cix < 0 or cix >= len(dp[ix - 1]):
                        continue
                    rs = min(rs, dp[ix - 1][cix])
                tmp_lst.append(rs + triangle[ix][jx])
            dp.append(tmp_lst)

        return min(dp[-1])


s = Solution()
triangle = [[2],
            [3,4],
            [6,5,7],
            [4,1,8,3]]
triangle = [[-10]]
print(s.minimumTotal(triangle))


