from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:

        rslt = []

        def run(now, candi):
            # print(sorted(now), rslt)
            if sorted(now) not in rslt:
                rslt.append(sorted(now))
            if not candi:
                return
            for c in candi:
                new_candi = [i for i in candi if i != c]
                run(now + [c], new_candi)
        run([], nums)
        return rslt


s = Solution()
nums = [4,1,0]
print(s.subsets(nums))

# print(sorted(nums))
