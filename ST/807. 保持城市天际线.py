from typing import List


class Solution:
    def maxIncreaseKeepingSkyline(self, grid: List[List[int]]) -> int:
        i_max = []
        j_max = []
        for ix in range(len(grid)):
            i_max.append(max(grid[ix]))
        for jx in range(len(grid[0])):
            j_max.append(max([grid[i][jx] for i in range(len(grid))]))

        rslt = [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        for ix in range(len(grid)):
            for jx in range(len(grid[0])):
                rslt[ix][jx] = min(i_max[ix], j_max[jx]) - grid[ix][jx]

        return sum([sum(rslt[ix]) for ix in range(len(rslt))])


s = Solution()
grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]
grid = [[0,0,0],[0,0,0],[0,0,0]]
print(s.maxIncreaseKeepingSkyline(grid))

