from typing import List

class Solution:
    def countBits(self, n: int) -> List[int]:
        rs = []
        for i in range(n + 1):
            rs.append(bin(i).count('1'))
        return rs


s = Solution()
print(s.countBits(5))