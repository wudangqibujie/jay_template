class Solution:
    def computeArea(self, ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
        if bx1 < ax1:
            ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 = bx1, by1, bx2, by2, ax1, ay1, ax2, ay2
        s1 = (bx2 - bx1) * (by2 - by1)
        s2 = (ax2 - ax1) * (ay2 - ay1)
        if ax2 <= bx1 or ay1 >= by2 or by1 >= ay2:
            return s1 + s2
        width = ax2 - bx1
        height = by2 - ay1 if ay2 >= by2 and by2 >= ay1 else ay2 - by1
        print(width, height, s1, s2)
        return s1 + s2 - width * height


s = Solution()
ax1 = -3
ay1 = 0
ax2 = 3
ay2 = 4
bx1 = 3
by1 = -1
bx2 = 9
by2 = 2


# ax1 = -2
# ay1 = -2
# ax2 = 2
# ay2 = 2
# bx1 = -2
# by1 = -2
# bx2 = 2
# by2 = 2
print(s.computeArea(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2))