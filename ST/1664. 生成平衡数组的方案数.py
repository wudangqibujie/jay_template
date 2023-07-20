from typing import List


class Solution:
    def waysToMakeFair(self, nums: List[int]) -> int:
        even_cum, odd_cum = [], []
        even, odd = 0, 0
        for ix in range(len(nums)):
            if ix % 2 == 0:
                even += nums[ix]
            else:
                odd += nums[ix]
            even_cum.append(even)
            odd_cum.append(odd)
        # print(even_cum)
        # print(odd_cum)
        cnt = 0
        for ix in range(len(nums)):
            if ix % 2== 0:
                even_val = even_cum[ix] - nums[ix] + odd_cum[-1] - odd_cum[ix]
                odd_val = odd_cum[ix] + even_cum[-1] - even_cum[ix]
            else:
                even_val = even_cum[ix] + odd_cum[-1] - odd_cum[ix]
                odd_val = odd_cum[ix] - nums[ix] + even_cum[-1] - even_cum[ix]
            if even_val == odd_val:
                cnt += 1
        return cnt



s = Solution()
nums = [2,1,6,4]
nums = [1, 1, 1]
nums = [1, 2, 3]
nums = [1, ]
print(s.waysToMakeFair(nums))