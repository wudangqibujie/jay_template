from typing import Optional
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def expandBinaryTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def helper(node):
            if node is None:
                return
            left = helper(node.left)
            right = helper(node.right)
            if left is not None:
                node.left = TreeNode(-1, left=left)
            if right is not None:
               node.right = TreeNode(-1, right=right)
            return node

        return helper(root)


s = Solution()
root = TreeNode(1, left=TreeNode(2, left=TreeNode(4), right=TreeNode(5)), right=TreeNode(3, left=TreeNode(6), right=TreeNode(7)))
root = s.expandBinaryTree(root)


def read(node):
    if node is None:
        return
    print(node.val)
    read(node.left)
    read(node.right)

read(root)
