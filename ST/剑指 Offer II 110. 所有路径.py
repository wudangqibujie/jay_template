from typing import List


class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        rslt = []
        def helper(route, now):
            if now == len(graph) - 1:
                rslt.append(route + [now])
                return
            for nx in graph[now]:
                helper(route + [now], nx)
        helper([], 0)
        return rslt


s = Solution()
graph = [[1,2],[3],[3],[]]
graph = [[4,3,1],[3,2,4],[3],[4],[]]
graph = [[1],[]]
graph = [[1,2,3],[2],[3],[]]
graph = [[1,3],[2],[3],[]]
print(s.allPathsSourceTarget(graph))