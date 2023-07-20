from typing import List

class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        rslt = []
        def search(now, candi):
            if len(now) == k:
                rslt.append(now)
                return

            for c in candi:
                new_can = [i for i in candi if i > c]
                search(now + [c], new_can)
        search([], list(range(1, n + 1)))
        return rslt


s = Solution()
n = 2
k = 2
print(s.combine(n, k))