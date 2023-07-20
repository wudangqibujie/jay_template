from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        l_max, r_max = [], []
        l_max = [height[0]]
        r_max = [height[-1]]
        for i in height[1: ]:
            l_max.append(max(l_max[-1], i))
        # print(l_max)

        ix = len(height) - 2
        while ix >= 0:
            r_max = [max(r_max[0], height[ix])] + r_max
            ix -= 1
        # print(r_max)

        rs = 0
        for ix in range(len(height)):
            rs += (min(r_max[ix], l_max[ix]) - height[ix])
        return rs


s = Solution()
height = [0,1,0,2,1,0,1,3,2,1,2,1]
height = [4,2,0,3,2,5]
height = [2, 0, 2]
print(s.trap(height))