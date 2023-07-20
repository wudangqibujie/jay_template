from typing import List


class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        dp = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        dp[0][0] = grid[0][0]
        for ix in range(1, len(grid)):
            dp[ix][0] = dp[ix - 1][0] + grid[ix][0]
        for jx in range(1, len(grid[0])):
            dp[0][jx] = dp[0][jx - 1] + grid[0][jx]
        for ix in range(1, len(grid)):
            for jx in range(1, len(grid[0])):
                dp[ix][jx] = min(dp[ix - 1][jx], dp[ix][jx - 1]) + grid[ix][jx]
        return dp[-1][-1]


s = Solution()
grid = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]
grid = [[1,2,3],[4,5,6]]
grid = [[
    1
]]
print(s.minPathSum(grid))