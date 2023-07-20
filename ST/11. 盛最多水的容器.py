from typing import List


class Solution:
    def ABC(self, height: List[int]) -> int:
        i, j = 0, len(height) - 1
        rslt = 0
        while i < j:
            rslt = max(rslt, (j - i) * min(height[i], height[j]))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return rslt

s = Solution()
height = [1,8,6,2,5,4,8,3,7]
height = [1, 1]
print(s.ABC(height))