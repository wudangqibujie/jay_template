class Solution:
    def partitionString(self, s: str) -> int:
        i, j = 0, 0
        cnt = 0
        readed = set()
        while j < len(s):
            if s[j] not in readed:
                readed.add(s[j])
                j += 1
            else:
                cnt += 1
                i = j
                readed = set()
        return cnt + 1


s = Solution()
S = 'a'
print(s.partitionString(S))