from typing import List


class Solution:
    def onesMinusZeros(self, grid: List[List[int]]) -> List[List[int]]:
        m, n = len(grid), len(grid[0])
        i_sum = [sum(i) for i in grid]
        j_sum = [sum(i) for i in zip(*grid)]
        i_zero = [m - i for i in i_sum]
        j_zero = [n - i for i in j_sum]
        diff = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                diff[i][j] = i_sum[i] + j_sum[j] - i_zero[i] - j_zero[j]
        # for i in diff:
        #     print(i)
        return diff

s = Solution()
grid = [[0,1,1],[1,0,1],[0,0,1]]
grid = [
    [1],
        # [1,1,1]
        ]
s.onesMinusZeros(grid)