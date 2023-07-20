from typing import List

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [[0 for _ in range(len(coins))] for _ in range(amount + 1)]
        for i in range(1, amount + 1):
            for j in range(len(coins)):
                lst_i = i - coins[j]

                if j == 0:
                    if dp[lst_i][j] == -1:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = dp[lst_i][j] + 1
                else:
                    if dp[lst_i][j] == -1:
                        if dp[lst_i][j] == -1:
                            dp[i][j] = -1
                        else:
                            dp[i][j] = dp[i][j - 1]
                    else:
                        dp[i][j] = dp[lst_i][j] + 1

        for i in dp:
            print(i)
        return dp[-1][-1]


s = Solution()
coins = [2, 3, 5]
amount = 11
print(s.coinChange(coins, amount))
