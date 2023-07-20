from typing import List
from copy import deepcopy
class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        rs = 0
        def search(route, candi):
            nonlocal rs
            if len(route) > 1:
                if route[-1] % route[-2] != 0 and route[-2] % route[-1] != 0:
                    return

            if not candi:
                if route[-1] % route[-2] == 0 or route[-2] % route[-1] == 0:
                    rs += 1
                return


            for c in candi:
                new_candi = [i for i in candi if i != c]
                search(route + [c], new_candi)
        search([], nums)
        return rs % (10 ** 9 + 7)


s = Solution()
nums = [1, 4, 3]
nums = [1,2,4,8,16,32,64]
print(s.specialPerm(nums))