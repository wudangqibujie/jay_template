import math

class Solution:
    def minOperations(self, n: int) -> int:
        val = 0
        ed = math.ceil(n / 2)
        for ix in range(ed):
            val += (n - (2 * ix + 1))
        return val


s = Solution()
print(s.minOperations(1))