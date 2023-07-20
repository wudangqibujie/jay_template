from typing import List

class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:

        map_ = dict()


        def search(nums):
            if len(nums) == 2:
                return [abs(nums[0] - nums[1])]
            if len(nums) == 1:
                return [nums[0]]
            tmp = []
            for ix in range(len(nums) - 1):
                for jx in range(ix + 1, len(nums)):
                    new_nums = [nums[kx] for kx in range(len(nums)) if kx not in (ix, jx)]
                    new_nums.sort()
                    sig = ','.join([str(i) for i in new_nums])
                    if sig in map_:
                        rs = map_[sig]
                    else:
                        rs = search(new_nums)
                    for i in rs:
                        val = abs(abs(nums[ix] - nums[jx]) - i)
                        if val not in tmp:
                            tmp.append(val)
            nums.sort()
            sig = ','.join([str(i) for i in nums])
            map_[sig] = tmp
            return tmp
        return min(search(stones))


s = Solution()
# stones = [2,7,4,1,8,1]
stones = [31,26,33,21,40, 31,26,33,21,40, 31,26,33,21,40, 31,26,33,21,40, 31,26,33,21,40, 31,26,33,21,40]
# stones = [1, 2]
# stones = [i for i in range(31)]
# print(s.lastStoneWeightII(stones))
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')
print(df.head())
print(df.country.unique())