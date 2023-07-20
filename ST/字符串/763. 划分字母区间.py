from typing import List

class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        mark = dict()
        for ix, i in enumerate(s):
            if i in mark:
                if len(mark[i]) == 1:
                    mark[i].append(ix)
                else:
                    mark[i][1] = ix
            else:
                mark[i] = [ix]
        mark = [i[1] for i in sorted(mark.items(), key=lambda x: x[1][0])]
        if len(mark[0]) == 1:
            rslt = [[i, i] for i in mark[0]][0]
        else:
            rslt = mark[0]

        rs = []
        for edge in mark[1:]:
            if len(edge) == 1:
                edge = [edge[0], edge[0]]
            if edge[0] > rslt[-1]:
                rs.append(rslt)
                rslt = edge
            else:
                rslt[1] = max(rslt[1], edge[1])
        rs.append(rslt)
        # print(rslt)
        # print(mark)
        # print(rs)
        return [i[1] + 1 - i[0] for i in rs]

s = Solution()
S = "caedbdedda"
print(s.partitionLabels(S))