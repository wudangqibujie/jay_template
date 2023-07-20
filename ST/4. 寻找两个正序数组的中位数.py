from typing import List

class Solution:
    def ABC(self, nums1: List[int], nums2: List[int]) -> float:

        target_idx = (len(nums1) + len(nums2)) // 2
        if (len(nums1) + len(nums2)) % 2 == 0:
            target_idx -= 1
        print(target_idx)

        def move_step(i, j):
            print(i, j)
            global lst_info
            if i == len(nums1):
                lst_info = [j, 'b']
                j += 1
                return i, j
            if j == len(nums2):
                lst_info = [i, 'a']
                i += 1
                return i, j
            if nums1[i] < nums2[j]:

                lst_info = [i, 'a']
                i += 1
            else:

                lst_info = [j, 'b']
                j += 1
            return i, j

        nxt_i, nxt_j = 0, 0
        for _ in range(target_idx + 1):
            nxt_i, nxt_j = move_step(nxt_i, nxt_j)
            # print(nxt_i, nxt_j, lst_info)
        idx, num = lst_info
        rslt = nums1[idx] if num == 'a' else nums2[idx]
        print(rslt)
        if (len(nums1) + len(nums2)) % 2 != 0:
            return rslt
        move_step(nxt_i, nxt_j)
        idx, num = lst_info
        print(lst_info)
        rslt += nums1[idx] if num == 'a' else nums2[idx]
        return rslt / 2

s = Solution()
nums1 = [1, 2, 3, 5]
nums2 = [4, 6, 7]

nums1 = [1,3]
nums2 = [2]

nums1 = [1,2]
nums2 = [3,4]

nums1 = []
nums2 = []

print(s.ABC(nums1, nums2))
