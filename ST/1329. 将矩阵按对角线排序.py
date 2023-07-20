from typing import List

class Solution:
    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        st_i, st_j = 0, 0
        while st_j < n:
            i, j = st_i, st_j
            rs = []
            while i < m and j < n:
                rs.append(mat[i][j])
                i += 1
                j += 1
            rs.sort()
            i, j = st_i, st_j
            for r_val in rs:
                mat[i][j] = r_val
                i += 1
                j += 1
            st_j += 1

        st_i, st_j = 1, 0
        while st_i < m:
            i, j = st_i, st_j
            rs = []
            while i < m and j < n:
                rs.append(mat[i][j])
                i += 1
                j += 1
            rs.sort()
            i, j = st_i, st_j
            for r_val in rs:
                mat[i][j] = r_val
                i += 1
                j += 1
            st_i += 1
        return mat


s = Solution()
mat = [
    [3],
]

rs = s.diagonalSort(mat)
for i in rs:
    print(i)
