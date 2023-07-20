from typing import List

class Solution:
    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        log = dict()
        for i in nums:
            if i in log:
                log[i] += 1
            else:
                log[i] = 1
        d = [(k, v) for k, v in log.items()]
        d = sorted(d, key=lambda x: x[1], reverse=True)
        res = [[] for _ in range(d[0][1])]
        for val, cnt in d:
            for ix in range(cnt):
                res[ix].append(val)
        return res

s = Solution()
nums = [1,3,4,1,2,3,1]
nums = [1,2,3,4]
nums = [1, ]
print(s.findMatrix(nums))