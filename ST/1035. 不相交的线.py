from typing import List


class Solution:
    def maxUncrossedLines(self, nums1: List[int], nums2: List[int]) -> int:
        dp = [[0 for _ in range(len(nums2) + 1)] for _ in range(len(nums1) + 1)]
        # for ix in range(len(nums1)):
        #     if str(nums1[ix]) in nums1[: ix]:
        #         nums1[ix] = f'{nums1[ix]}:{nums1[: ix].count(nums1[ix])}'
        #     else:
        #         nums1[ix] = str(nums1[ix])
        #
        # for ix in range(len(nums2)):
        #     if str(nums2[ix]) in nums2[: ix]:
        #         nums2[ix] = f'{nums2[ix]}:{nums2[: ix].count(nums2[ix])}'
        #     else:
        #         nums2[ix] = str(nums2[ix])

        for ix in range(len(nums1)):
            for jx in range(len(nums2)):
                if nums1[ix] == nums2[jx]:
                    dp[ix + 1][jx + 1] = dp[ix][jx] + 1
                else:
                    dp[ix + 1][jx + 1] = max(dp[ix][jx + 1], dp[ix + 1][jx])
        print(nums1)
        print(nums2)
        for i in dp:
            print(i)
        return dp[-1][-1]
s = Solution()
nums1 = [2,5,1,2,5]
nums2 = [10,5,2,1,5,2]
# nums1 = [1,3,7,1,7,5]
# nums2 = [1,9,2,5,1]
# nums1 = [3]
# nums2 = [0, 4]
#
# nums1 = [2,1]
# nums2 = [1,2,1,3,3,2]
print(s.maxUncrossedLines(nums1, nums2))