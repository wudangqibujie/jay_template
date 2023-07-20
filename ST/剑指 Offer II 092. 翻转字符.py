class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        dp = [(int(s[0] != '0'), int(s[0] != '1'))]
        for ix in range(1, len(s)):
            tmp1 = dp[ix - 1][0] + int(s[ix] != '0')
            tmp2 = min(dp[-1]) + int(s[ix] != '1')
            dp.append((tmp1, tmp2))
        return min(dp[-1])

s = Solution()
print(s.minFlipsMonoIncr("101"))

