class Solution:
    def isFascinating(self, n: int) -> bool:
        rs = str(n) + str(n * 2) + str(n * 3)
        if '0' in rs:
            return False
        if len(set(rs)) != 9:
            return False
        if len(rs) != 9:
            return False
        return True


s = Solution()
n = 783
print(s.isFascinating(n))