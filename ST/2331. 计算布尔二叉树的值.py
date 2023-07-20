# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from typing import Optional
class Solution:
    def evaluateTree(self, root: Optional[TreeNode]) -> bool:


        def read(node):
            if node.val in [0, 1]:
                return node.val == 1
            left_val = read(node.left)
            right_val = read(node.right)
            if node.val == 2:
                return left_val or right_val
            else:
                return left_val and right_val


        return read(root)


s = Solution()
tree = TreeNode(2)
tree.left = TreeNode(1)
tree.right = TreeNode(3)
tree.right.left = TreeNode(0)
tree.right.right = TreeNode(0)
tree = TreeNode(0)
print(s.evaluateTree(tree))


