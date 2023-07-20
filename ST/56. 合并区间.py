from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: (x[0], x[1]))
        # print(intervals)
        rslt = [intervals[0]]
        for interval in intervals[1: ]:
            if interval[0] > rslt[-1][1]:
                rslt.append(interval)
            else:
                st = rslt[-1][0]
                ed = max(interval[1], rslt[-1][1])
                rslt[-1] = [st, ed]
        return rslt



s = Solution()
intervals = [[1,3],[2,6],[8,10],[15,18]]
intervals = [[1,4],[4,5]]
intervals = [[1, 2], [0, 3]]
import random
random.shuffle(intervals)
print(s.merge(intervals))