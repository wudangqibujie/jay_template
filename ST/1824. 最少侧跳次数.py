from typing import List


class Solution:
    def minSideJumps(self, obstacles: List[int]) -> int:
        dp = [[0 for _ in range(len(obstacles))] for _ in range(3)]
        dp[0][0] = 1
        dp[1][0] = 0
        dp[2][0] = 1

        for ix in range(3):
            for jx in range(1, len(obstacles)):
                if ix == obstacles[jx] - 1:
                    dp[ix][jx] = float('inf')
                    continue
        for jx in range(1, len(obstacles)):
            for ix in range(3):
                candi = [dp[i][jx - 1] + 1 for i in range(3)]
                for c in range(3):
                    if dp[c][jx] == float('inf'):
                        candi[c] = float('inf')
                candi[ix] -= 1
                # print(ix, jx, candi)
                dp[ix][jx] = max(min(candi), dp[ix][jx])
        # for i in dp:
        #     print(i)
        return min([dp[i][-1] for i in range(3)])


s = Solution()
obstacles = [0,1,2,3,0]
obstacles = [0,1,1,3,3,0]
obstacles = [0,2,1,0,3,0]
print(s.minSideJumps(obstacles))