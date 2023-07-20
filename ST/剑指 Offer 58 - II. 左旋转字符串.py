class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        n = n % len(s)
        return s[n: ] + s[: n]


so = Solution()
s = "abcdefg"
k = 2
s = "lrloseumgh"
k = 6
print(so.reverseLeftWords(s, k))

