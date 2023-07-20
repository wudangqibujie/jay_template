class Solution:
    def ABC(self, s: str, p: str) -> bool:
        dp = [[0 for _ in range(len(s) + 1)] for _ in range(len(p) + 1)]

        s = ' ' + s
        p = ' ' + p
        dp[0][0] = 1

        for i in range(1, len(p)):
            if p[i] == '*':
                dp[i][0] = dp[i - 2][0]

        for i in range(1, len(p)):
            for j in range(1, len(s)):
                if p[i] == '.':
                    dp[i][j] = 0 if dp[i - 1][j - 1] == 0 else 1
                elif p[i] == '*':
                    if (s[j] == p[i - 1] or p[i - 1] == '.'):
                        dp[i][j] = 1
                else:
                    if p[i] != s[j]:
                        dp[i][j] = 0
                    else:
                        dp[i][j] = 1 if dp[i - 1][j - 1] == 1 else 0

        for i in dp:
            print(i)
        return dp[-1][-1] == 1


so = Solution()
s = "mississippi"
p = "mis*is*p*."

s = "mississipp"
p = "mis*is*ip*"

s = 'aab'
p = 'c*a*b'

s = 'c'
p = 'c*'

# s = 'aaa'
# p = 'ab*ac*a'
#
s = 'a'
p = 'ab*'

print(so.ABC(s, p))




