from typing import List
import heapq

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        stones = [-i for i in stones]
        heapq.heapify(stones)
        print(stones)
        print(heapq.heappop(stones))
        print(stones)
        heapq.heappush(stones, 9)
        print(stones)
        while True:
            print(heapq.heappop(stones))


s = Solution()
stones = [2,7,4,1,8,1]
s.lastStoneWeight(stones)