class Solution:
    def countSubstrings(self, s: str, t: str) -> int:
        dp = [[0 for _ in range(len(s))] for _ in range(len(t))]
        for ix in range(len(t)):
            if s[0] != t[ix]:
                dp[ix][0] = 1
        for jx in range(len(s)):
            if t[0] != s[jx]:
                dp[0][jx] = 1

        for ix in range(1, len(t)):
            for jx in range(1, len(s)):
                res = 0
                if t[ix] != s[jx]:
                    res += 1
                    st_i, st_j = ix - 1, jx - 1
                    while st_i >= 0 and st_j >= 0:
                        if t[st_i] == s[st_j]:
                            res += 1
                        else:
                            break
                        st_i -= 1
                        st_j -= 1
                else:
                    st_i, st_j = ix - 1, jx - 1
                    while st_i >= 0 and st_j >= 0 and t[st_i] == s[st_j]:
                        st_i -= 1
                        st_j -= 1

                    if st_i >= 0 and st_j >= 0 and t[st_i] != s[st_j]:
                        res += 1
                        st_i -= 1
                        st_j -= 1
                    else:
                        continue
                    while st_i >= 0 and st_j >= 0:
                        if t[st_i] == s[st_j]:
                            res += 1
                        else:
                            break
                        st_i -= 1
                        st_j -= 1
                dp[ix][jx] = res
        for i in dp:
            print(i)
        return sum([sum(i) for i in dp])




s = Solution()
S = "abbab"
T = "bbbbb"
print(s.countSubstrings(S, T))