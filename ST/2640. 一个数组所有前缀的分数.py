from typing import List


class Solution:
    def findPrefixScore(self, nums: List[int]) -> List[int]:
        convert = []
        tmp_max = nums[0]
        for n in nums:
            tmp_max = max(tmp_max, n)
            convert.append(tmp_max + n)
        # print(convert)
        rslt = [convert[0]]
        for i in convert[1:]:
            rslt.append(rslt[-1] + i)
        return rslt


s = Solution()
nums = [2,3,7,5,10]
# nums = [1,1,2,4,8,16]
nums = [2,]
print(s.findPrefixScore(nums))