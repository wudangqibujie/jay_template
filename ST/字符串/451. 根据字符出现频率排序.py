class Solution:
    def frequencySort(self, s: str) -> str:
        log = dict()
        for c in s:
            if c not in log:
                log[c] = 1
            else:
                log[c] += 1
        log = sorted(log.items(), key=lambda x: x[1], reverse=True)
        return ''.join([i[0] * i[1] for i in log])


s = Solution()  
S = "Aabb"
print(s.frequencySort(S))