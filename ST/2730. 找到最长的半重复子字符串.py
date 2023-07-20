class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        if len(s) == 1:
            return 1
        rs = float('-inf')
        status = []
        i, j = 0, 1
        while j < len(s) and i <= j:
            if s[j] == s[j - 1]:
                if status:
                    i = status.pop()
                    status.append(j)
                else:
                    status.append(j)
            j += 1
            rs = max(rs, j - i)
        return rs


s = Solution()
string = "11"
print(s.longestSemiRepetitiveSubstring(string))