from typing import List


class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        readed = [[None for _ in range(len(grid[0]))] for _ in range(len(grid))]

        for i in readed:
            print(i)

        def find(i, j):
            if i < 0 or i > len(grid) - 1 or j < 0 or j > len(grid[0]) - 1:
                return False
            if grid[i][j] == 1:
                return True
            candi = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            rslt = []
            for ni, nj in candi:
                if readed[ni][nj] is None:
                    rs = find(ni, nj)
                else:
                    rs = readed[ni][nj]
                rslt.append(rs)
            if False in rslt:
                return False
            return True

        rslt = find(1, 1)
        print(rslt)

s = Solution()
grid = [[1,1,1,1,1,1,1,0],
        [1,0,0,0,0,1,1,0],
        [1,0,1,0,1,1,1,0],
        [1,0,0,0,0,1,0,1],
        [1,1,1,1,1,1,1,0]]
s.closedIsland(grid)