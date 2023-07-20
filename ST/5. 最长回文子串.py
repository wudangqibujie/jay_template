class Solution:
    def ABC(self, s: str) -> str:
        rslt = 1
        rslt_s = s[0]
        def search(st, ed):
            nonlocal rslt_s, rslt
            while st >= 0 and ed < len(s):
                if s[st] != s[ed]:
                    break
                if ed - st + 1 > rslt:
                    rslt_s = s[st: ed + 1]
                rslt = max(rslt, ed - st + 1)
                st -= 1
                ed += 1
            return rslt

        for ix in range(len(s)):
            rslt = max(rslt, search(ix, ix + 1))
            rslt = max(rslt, search(ix - 1, ix + 1))
        return rslt_s


s = Solution()
print(s.ABC("abaaa  a"))