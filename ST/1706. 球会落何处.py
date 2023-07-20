from typing import List


class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        m, n = len(grid), len(grid[0])
        rs = []
        for jx in range(n):
            for ix in range(m):
                if grid[ix][jx] == 1:
                    jx += 1
                    if jx >= n or grid[ix][jx] == -1:
                        break
                else:
                    jx -= 1
                    if jx < 0 or grid[ix][jx] == 1:
                        break
            if ix == m - 1:
                if jx >= 0 and jx < n:
                    rs.append(jx)
                else:
                    rs.append(-1)
            else:
                rs.append(-1)
        return rs


s = Solution()
grid = [[1,1,1,-1,-1],
        [1,1,1,-1,-1],
        [-1,-1,-1,1,1],
        [1,1,1,1,-1],
        [-1,-1,-1,-1,-1]]
grid = [[-1]]
grid = [[1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1],[1,1,1,1,1,1],[-1,-1,-1,-1,-1,-1]]
grid = [[1]]
rs = s.findBall(grid)
print(rs)