from typing import List


class Solution:
    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        log = dict()
        for ix, i in enumerate(groupSizes):
            if i not in log:
                log[i] = [ix]
            else:
                log[i].append(ix)
        rslt = []
        for k, v in log.items():
            for b_n in range(len(v) // k):
                st, ed = k * b_n, k * (b_n + 1)
                rslt.append(v[st: ed])

        return rslt


s = Solution()
groupSizes = [3,3,3,3,3,1,3]
groupSizes = [2,1,3,3,3,2]
print(s.groupThePeople(groupSizes))