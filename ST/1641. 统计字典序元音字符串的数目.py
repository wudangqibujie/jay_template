

class Solution:
    def countVowelStrings(self, n: int) -> int:
        n = n + 1
        dp = [[1 for _ in range(n)] for _ in range(5)]
        for i in range(5):
            dp[i][0] = 0
        for j in range(1, n):
            for i in range(1, 5):
                dp[i][j] = dp[i][j-1] + dp[i-1][j]
        # for i in dp:
        #     print(i)

        return sum(dp[i][-1] for i in range(5))


s = Solution()
print(s.countVowelStrings(33))