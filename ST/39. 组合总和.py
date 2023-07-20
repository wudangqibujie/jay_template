from typing import List

class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        dp = [[[]]]

        for t in range(1, target + 1):
            rslt = []
            for c in candidates:
                left = t - c
                if left < 0:
                    continue
                # print(t, c, left, rslt, dp)
                for l in dp[left]:
                    bu = l + [c]
                    bu.sort()
                    if bu not in rslt:
                        rslt.append(bu)
            dp.append(rslt)
        return dp[-1]


s = Solution()
candidates = [2,3,6,7]
target = 7

candidates = [2,3,5]
target = 8

candidates = [2]
target = 1
print(s.combinationSum(candidates, target))