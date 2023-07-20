from typing import List

class Solution:
    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        S = set()
        for e in edges:
            S.add(e[1])
        rslt = []
        for i in range(n):
            if i not in S:
                rslt.append(i)
        return rslt


s = Solution()
n = 5
edges = [[0,1],[2,1],[3,1],[1,4],[2,4]]
n = 6
edges = [[0,1],[0,2],[2,5],[3,4],[4,2]]
n = 4
edges = [[0, 1], [1, 2], [3, 1]]
print(s.findSmallestSetOfVertices(n, edges))