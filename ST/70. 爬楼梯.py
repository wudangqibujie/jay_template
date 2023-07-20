class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:return 1
        if n == 2: return 2
        dp = [1, 2]
        for _ in range(2, n):
            dp.append(dp[-1] + dp[-2])
        return dp[-1]


s = Solution()
print(s.climbStairs(3))