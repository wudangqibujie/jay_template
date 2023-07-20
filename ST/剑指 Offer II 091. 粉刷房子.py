from typing import List

class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        dp = [[0 for _ in range(len(costs[0]))] for _ in range(len(costs))]
        dp[0] = costs[0]
        for i in range(1, len(costs)):
            for j in range(3):
                dp[i][j] = min([dp[i - 1][jx] for jx in range(3) if jx != j]) + costs[i][j]
        # for i in dp:
        #     print(i)
        return min(dp[-1])


s = Solution()
costs = [[17,2,17],
         [16,16,5],
         [14,3,19]]
costs = [[7,6,2]]
print(s.minCost(costs))