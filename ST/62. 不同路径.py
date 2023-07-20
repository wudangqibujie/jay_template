

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for ix in range(1, m):
            for jx in range(1, n):
                dp[ix][jx] = dp[ix - 1][jx] + dp[ix][jx - 1]
        return dp[-1][-1]


s = Solution()
m = 1

n = 10
print(s.uniquePaths(m, n))