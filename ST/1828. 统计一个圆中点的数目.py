from typing import List


class Solution:
    def countPoints(self, points: List[List[int]], queries: List[List[int]]) -> List[int]:
        rslt = []
        for x, y, r in queries:
            bu = 0
            for px, py in points:
                if (x - px) ** 2 + (y - py) ** 2 <= r ** 2:
                    bu += 1
            rslt.append(bu)
        return rslt


s = Solution()
points = [[1,3],[3,3],[5,3],[2,2]]

points = [[1,1],[2,2],[3,3],[4,4],[5,5]]
queries = [[1,2,2],[2,2,2],[4,3,2],[4,3,3]]

print(s.countPoints(points, queries))