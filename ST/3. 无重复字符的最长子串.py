class Solution:
    def ABC(self, s: str) -> int:
        if not s:
            return 0
        i, j = 0, 1
        log = dict()
        log[s[0]] = 0
        rslt = 1
        while j < len(s):
            if s[j] not in log:
                log[s[j]] = j
                rslt = max(rslt, len(log))
                j += 1
                continue
            nxt_idx = log[s[j]]
            for x in range(i, nxt_idx + 1):
                log.pop(s[x])
            i = nxt_idx + 1
        return rslt


s = Solution()
print(s.ABC("pwwkew"))