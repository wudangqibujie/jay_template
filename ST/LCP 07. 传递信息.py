from typing import List


class Solution:
    def numWays(self, n: int, relation: List[List[int]], k: int) -> int:
        log = dict()
        for edge in relation:
            if edge[0] in log:
                log[edge[0]].append(edge[1])
            else:
                log[edge[0]] = [edge[1]]
        print(log)
        def search(node):
            if node == n - 1:
                return [[n - 1]]
            if node not in log:
                return [[]]
            rslt = []
            for ca in log[node]:
                rs = search(ca)
                if rs[0]:
                    for r in rs:
                        rslt.append([node] + r)
            print(node, rslt)
            return rslt

        r = search(0)
        for r in r:
            print(r)


s = Solution()
n = 5
relation = [[0,2],[2,1],[3,4],[2,3],[1,4],[2,0],[0,4]]
k = 3
print(s.numWays(n, relation, k))