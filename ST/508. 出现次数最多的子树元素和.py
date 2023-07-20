# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
from typing import Optional, List

class Solution:
    def findFrequentTreeSum(self, root: Optional[TreeNode]) -> List[int]:
        rs = dict()
        def search(node):
            if node is None:
                return 0
            left = search(node.left)
            right = search(node.right)
            sum_val = left + right + node.val
            if sum_val not in rs:
                rs[sum_val] = 1
            else:
                rs[sum_val] += 1
            return sum_val

        search(root)

        max_cnt = max(rs.values())
        rslt = []
        for k, v in rs.items():
            if v == max_cnt:
                rslt.append(k)
        return rslt


s = Solution()

tree = TreeNode(5)
# tree.left = TreeNode(5)
# tree.right = TreeNode(-5)
print(s.findFrequentTreeSum(tree))