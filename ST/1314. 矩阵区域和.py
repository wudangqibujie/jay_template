from typing import List

class Solution:
    def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        cumsum_matrix = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cumsum_matrix[i][j] = cumsum_matrix[i - 1][j] + cumsum_matrix[i][j - 1] - cumsum_matrix[i - 1][j - 1] + mat[i - 1][j - 1]
        cumsum_matrix = [i[1: ] for i in cumsum_matrix[1: ]]
        for i in cumsum_matrix:
            print(i)

        res = [[0 for _ in range(n)] for _ in range(m)]
        for ix in range(m):
            for jx in range(n):
                

                i1, j1 = min(ix + k, m- 1), min(jx + k, n - 1)
                s1 = cumsum_matrix[i1][j1]

                i2, j2 = max(0, ix - k - 1), min(jx + k, n - 1)
                s2 = cumsum_matrix[i2][j2]

                i3, j3 = max(ix - k - 1, 0), min(jx - k - 1, n - 1)
                s3 = cumsum_matrix[i3][j3]

                i4, j4 = max(ix + k - 1, 0), max(jx - k - 1, 0)
                s4 = cumsum_matrix[i4][j4]

                res[ix][jx] = s1 - s2 - s3 + s4
        print()
        for i in res:
            print(i)



s = Solution()
matrix = [[1,2,3],[4,5,6],[7,8,9]]
s.matrixBlockSum(matrix, 1)