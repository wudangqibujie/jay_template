from typing import List


class Solution:
    def maxWidthOfVerticalArea(self, points: List[List[int]]) -> int:
        points.sort()
        rs = 0
        for ix in range(1, len(points)):
            rs = max(rs, points[ix][0] - points[ix - 1][0])
        return rs


s = Solution()
points = [[8,7],[9,9],[7,4],[9,7]]
points = [[3,1],[9,0],[1,0],[1,4],[5,3],[8,8]]

print(s.maxWidthOfVerticalArea(points))