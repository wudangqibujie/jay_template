class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if not s and p == '*':
            return True
        dp = [[0 for _ in range(len(s) + 1)] for _ in range(len(p) + 1)]
        dp[0][0] = 1
        for ix in range(len(p)):
            if p[ix] == '*' and dp[ix][0] == 1:
                dp[ix + 1][0] = 1
        for ix in range(len(p)):
            for jx in range(len(s)):
                if p[ix] == '*':
                    dp[ix + 1][jx + 1] = max(dp[ix + 1][jx], dp[ix][jx + 1], dp[ix][jx])
                elif p[ix] == '?':
                    dp[ix + 1][jx + 1] = dp[ix][jx]
                else:
                    if p[ix] == s[jx]:
                        dp[ix + 1][jx + 1] = dp[ix][jx]
                    else:
                        dp[ix + 1][jx + 1] = 0
        for i in dp:
            print(i)
        return dp[-1][-1] == 1


so = Solution()
s = ""
p = "?"
s = "adceb"
p = "*a*b"
# s = "aab"
# p = "c*a*b"
print(so.isMatch(s, p))
