from typing import List

class Solution:
    def matrixScore(self, grid: List[List[int]]) -> int:
        def sum_grid():
            return sum([int(f'0b{"".join([str(i) for i in nb])}', 2) for nb in grid])
        def flip_row(ix):
            for jx in range(len(grid[0])):
                grid[ix][jx] = 1 - grid[ix][jx]
        def flip_col(jx):
            for ix in range(len(grid)):
                grid[ix][jx] = 1 - grid[ix][jx]

        for ix in range(len(grid)):
            val1 = sum_grid()
            flip_row(ix)
            val2 = sum_grid()
            if val2 > val1:
                continue
            else:
                flip_row(ix)

        for jx in range(len(grid[0])):
            val1 = sum_grid()
            flip_col(jx)
            val2 = sum_grid()
            if val2 > val1:
                continue
            else:
                flip_col(jx)
        return sum_grid()

grid = [[0,0,1,1],[1,0,1,0],[1,1,0,0]]
grid = [[0]]
s = Solution()
print(s.matrixScore(grid))