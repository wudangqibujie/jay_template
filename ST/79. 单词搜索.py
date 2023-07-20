from typing import List


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:


        def find(ix, i, j, readed):
            readed.append((i, j))
            if ix == len(word):
                return True
            if board[i][j] != word[ix]:
                return False
            rslt = []
            candi = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            for nxt_i, nxt_j in candi:
                if nxt_i < 0 or nxt_i >= len(board) or nxt_j < 0 or nxt_j >= len(board[0]) or (nxt_i, nxt_j) in readed:
                    continue
                r = find(ix+1, nxt_i, nxt_j)
                print(i, j, r, nxt_i, nxt_j)
                rslt.append(r)
            print(i, j, rslt)
            if sum(rslt) == 0:
                return False
            return True

        for i in range(len(board)):
            for j in range(len(board[0])):
                readed = []
                print('*************************')
                if find(0, i, j):
                    return True
        return False

s = Solution()
board = [["A","B","C","E"],
         ["S","F","C","S"],
         ["A","D","E","E"]]
word = "ABFS"
print(s.exist(board, word))