from typing import List


class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        tmp_map = dict()
        readed = set()
        def search(i, j):
            if i < 0 or j < 0 or i >= len(mat) or j >= len(mat[0]):
                return []
            if mat[i][j] == 1:

                return [(i, j)]
            if (i, j) in tmp_map:
                return tmp_map[(i, j)]
            candi_idxes = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
            rs = []
            for can_i , can_j in candi_idxes:
                tmp_rs = search(can_i, can_j)
                rs.extend(tmp_rs)
            tmp_map[(i, j)] = rs
            return rs

        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] == 0:
                    rs = search(i, j)
                    print(rs)
                    break


s = Solution()
mat = [[0,0,0],
       [0,1,0],
       [0,0,0]]
mat = [[0,0,0],
       [0,1,0],
       [1,1,1]]
print(s.updateMatrix(mat))
