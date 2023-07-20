# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from typing import Optional
class Solution:
    def averageOfSubtree(self, root: Optional[TreeNode]) -> int:

        rslt = 0
        def helper(node):
            nonlocal rslt
            if node is None:
                return 0, 0
            left_val, left_cnt = helper(node.left)
            right_val, right_cnt = helper(node.right)
            if (left_val + right_val + node.val) // (left_cnt + right_cnt + 1) == (node.val):
                rslt += 1
            return left_val + right_val + node.val, left_cnt + right_cnt + 1
        helper(root)
        return rslt


s = Solution()
tree = TreeNode(4)
# tree.left = TreeNode(8)
# tree.right = TreeNode(5)
# tree.left.left = TreeNode(0)
# tree.left.right = TreeNode(1)
# tree.right.right = TreeNode(6)
print(s.averageOfSubtree(tree))