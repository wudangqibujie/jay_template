from typing import List

class Solution:
    def differenceOfDistinctValues(self, grid: List[List[int]]) -> List[List[int]]:
        m, n = len(grid), len(grid[0])
        i, j = 0, 0
        ans = [[0 for _ in range(n)] for _ in range(m)]
        while j < n and i < m:
            tmp_lst = []
            ix, jx = i, j
            while ix < m and jx < n:
                tmp_lst.append(grid[ix][jx])
                ix += 1
                jx += 1
                if ix >= m or jx >= n:
                    break
            ix, jx = i, j
            tmp_ix = 0
            while ix < m and jx < n:
                l_len = len(set(tmp_lst[: tmp_ix]))
                r_len = len(set(tmp_lst[tmp_ix + 1:]))
                ans[ix][jx] = abs(l_len - r_len)
                ix += 1
                jx += 1
                tmp_ix += 1
                if ix >= m or jx >= n:
                    break
            j += 1

        i, j = 1, 0
        while j < n and i < m:
            tmp_lst = []
            ix, jx = i, j
            while ix < m and jx < n:
                tmp_lst.append(grid[ix][jx])
                ix += 1
                jx += 1
                if ix >= m or jx >= n:
                    break
            ix, jx = i, j
            tmp_ix = 0
            while ix < m and jx < n:
                l_len = len(set(tmp_lst[: tmp_ix]))
                r_len = len(set(tmp_lst[tmp_ix + 1:]))
                ans[ix][jx] = abs(l_len - r_len)
                ix += 1
                jx += 1
                tmp_ix += 1
                if ix >= m or jx >= n:
                    break
            i += 1

        return ans


s = Solution()
# grid = [[1,2,3],
#         [4,5,6],
#         [7,8,9]]
grid = [[1,1,1],
        [3,4,5],
        [6,7,8]]
# grid = [[1,2,3,4],
#         [5,6,7,8]]
# grid = [[1,2,3],[3,1,5],[3,2,1]]
# grid = [[1]]
s.differenceOfDistinctValues(grid)

