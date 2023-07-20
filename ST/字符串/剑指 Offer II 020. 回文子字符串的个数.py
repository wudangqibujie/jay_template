class Solution:
    def countSubstrings(self, s: str) -> int:
        cnt = 0
        def check(i, j):
            ix, jx = i, j - 1
            while ix <= jx:
                if s[ix] != s[jx]:
                    return False
                ix += 1
                jx -= 1
            return True

        for i in range(len(s)):
            for j in range(i + 1, 1 + len(s)):
                if check(i, j):
                    cnt += 1
        return cnt

s = Solution()
ss = 'aaa'
print(s.countSubstrings(ss))