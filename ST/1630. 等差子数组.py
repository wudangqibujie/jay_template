from typing import List

class Solution:
    def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
        rslt = []
        for ix in range(len(l)):
            l_ix, r_ix = l[ix], r[ix] + 1
            sub_lst = nums[l_ix: r_ix]
            sub_lst.sort()
            if len(sub_lst) < 2:
                rslt.append(False)
                continue
            diff = sub_lst[1] - sub_lst[0]
            kx = 1
            while kx < len(sub_lst):
                if sub_lst[kx] - sub_lst[kx - 1] != diff:
                    rslt.append(False)
                    break
                kx += 1
            if kx == len(sub_lst):
                rslt.append(True)
        return rslt


s = Solution()
nums = [4,6,5,9,3,7]
l = [0,0,2]
r = [2,3,5]
nums = [-12,-9,-3,-12,-6,15,20,-25,-20,-15,-10]
l = [0,1,6,4,8,7]
r = [4,4,9,7,9,10]

print(s.checkArithmeticSubarrays(nums, l, r))