# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
from typing import Optional, List
class Solution:
    def printTree(self, root: Optional[TreeNode]) -> List[List[str]]:
        def cal_height(node):
            if node is None:
                return 0
            left_height = cal_height(node.left)
            right_height = cal_height(node.right)
            return max(left_height, right_height) + 1

        height = cal_height(root)
        rs = [[''] * (2 ** height - 1) for _ in range(height)]
        for i in rs:
            print(i)


s = Solution()
tree = TreeNode(1)
tree.left = TreeNode(2)
tree.right = TreeNode(3)
tree.left.right = TreeNode(4)
print(s.printTree(tree))