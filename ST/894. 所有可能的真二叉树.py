from typing import Optional, List

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def allPossibleFBT(self, n: int) -> List[Optional[TreeNode]]:
        if n % 2 == 0:
            return []
        msp_ = {1: [TreeNode(0)]}
        if n == 1:
            return msp_[1]
        for ix in range(3, n + 1, 2):
            needed_pair = []
            st, stop = 1, ix - 1
            while st < stop:
                needed_pair.append((st, stop - st))
                st += 2
            tmp_contain = []
            for pair in needed_pair:
                left_choice = msp_[pair[0]]
                right_choice = msp_[pair[1]]
                for l in left_choice:
                    for r in right_choice:
                        tmp_contain.append(TreeNode(0, l, r))
            msp_[ix] = tmp_contain
        return msp_[n]


s = Solution()
print(s.allPossibleFBT(21))
