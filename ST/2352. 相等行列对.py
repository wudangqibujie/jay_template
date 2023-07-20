from typing import List

class Solution:
    def ABC(self, grid: List[List[int]]) -> int:
        cnt = 0
        i = 0
        def check_rslt(i, j):
            a, b = 0, 0
            while a < len(grid):
                if grid[i][b] == grid[a][j]:
                    b += 1
                    a += 1
                else:
                    return False
            return True

        while i < len(grid):
            j = 0
            while j < len(grid[0]):
                rslt = check_rslt(i, j)
                # print(i, j, rslt)
                cnt += rslt
                j += 1
            i += 1

        return cnt


s = Solution()
grid = [[3,1,2,2],[1,4,4,5],[2,4,2,2],[2,4,2,2]]
print(s.ABC(grid))