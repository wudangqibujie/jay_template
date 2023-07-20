from typing import List

class Solution:
    def processQueries(self, queries: List[int], m: int) -> List[int]:
        rs = []
        P = [i for i in range(1, m + 1)]
        for ix, i in enumerate(queries):
            rs_ix = P.index(i)
            rs.append(rs_ix)
            while rs_ix > 0:
                P[rs_ix], P[rs_ix - 1] = P[rs_ix - 1], P[rs_ix]
                rs_ix -= 1
        return rs


s = Solution()
queries = [3,1,2,1]
m = 5
queries = [4,1,2,2]
m = 4
queries = [7,5,5,8,3]
m = 8
print(s.processQueries(queries, m))
