import heapq
from typing import List
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        rslt = []
        nums = [-i for i in nums]
        list_k = nums[: k]
        heapq.heapify(list_k)
        rslt.append(-list_k[0])
        for i in range(1, len(nums) - k + 1):
            print(nums[i: i + k], nums[i - 1], list_k)
            if nums[i - 1] == list_k[0]:
                heapq.heappop(list_k)
            heapq.heappush(list_k, nums[i + k - 1])
            rslt.append(-list_k[0])
        return rslt
so = Solution()
nums = [9,10,9,-7,-4,-8,2,-6]
# k = 3
#
# nums = [1]
k = 5
print(so.maxSlidingWindow(nums, k))