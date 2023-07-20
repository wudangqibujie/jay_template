from typing import List


class Solution:
    def hIndex(self, citations: List[int]) -> int:
        i, j = 0, len(citations)
        rs = 0
        while i < j:
            mid = (j - i) // 2 + i
            if citations[mid] > len(citations) - mid:
                j = mid
            else:
                i = mid + 1
                rs = max(rs, citations[mid])
        return rs


s = Solution()
citations = [0,1,3,5,6]
# citations = [1,2,100]
citations = [0, 1, 1, 1, 2, 3, 3, 4, 4, 33, 55]
citations = [10]
print(s.hIndex(citations))