
from typing import  List
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        j_st, j_ed = 0, len(matrix) - 1
        while j_st < j_ed:
            for i in range(len(matrix)):
                matrix[i][j_st], matrix[i][j_ed] = matrix[i][j_ed], matrix[i][j_st]
            j_st += 1
            j_ed -= 1
        # for i in matrix:
        #     print(i)
        # print()
        for jx in range(len(matrix)):
            st_i, st_j = 0, jx
            ed_i, ed_j = len(matrix) - 1 - jx, len(matrix) - 1
            while st_i < ed_i:
                matrix[st_i][st_j], matrix[ed_i][ed_j] = matrix[ed_i][ed_j], matrix[st_i][st_j]
                st_i += 1
                ed_i -= 1
                st_j += 1
                ed_j -= 1
        # for i in matrix:
        #     print(i)
        for jx in range(1, len(matrix)):
            st_i, st_j = jx, 0
            ed_i, ed_j = len(matrix) - 1, len(matrix) - jx - 1
            while st_i < ed_i:
                matrix[st_i][st_j], matrix[ed_i][ed_j] = matrix[ed_i][ed_j], matrix[st_i][st_j]
                st_i += 1
                ed_i -= 1
                st_j += 1
                ed_j -= 1

        # print()
        # for i in matrix:
        #     print(i)



s = Solution()
matrix = [[1,2,3],
          [4,5,6],
          [7,8,9]]
# matrix = [[1, 2],
#           [3, 4]]
# matrix = [[1]]
# matrix = [[5,1,9,11],
#           [2,4,8,10],
#           [13,3,6,7],
#           [15,14,12,16]]

s.rotate(matrix)
