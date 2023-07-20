from typing import List

class Solution:
    def minOperations(self, boxes: str) -> List[int]:
        candi = []
        for ix, i in enumerate(boxes):
            if i == '1':
                candi.append(ix)
        rslt = []
        for ix, i in enumerate(boxes):
            bu = 0
            for cix, cin in enumerate(candi):
                if ix != cin:
                    bu += abs(ix - cin)
            rslt.append(bu)
        return rslt


s = Solution()
boxes = "001011"
print(s.minOperations(boxes))