from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:

        rslt = []
        def run(now, candi):
            if not candi:
                rslt.append(now)
                return
            for c in candi:
                new_candi = [i for i in candi if i != c]
                run(now + [c], new_candi)

        run([], nums)
        return rslt


s = Solution()
nums = [1, 2, 3]
nums = [0, 1]
nums = [1]
print(s.permute(nums))