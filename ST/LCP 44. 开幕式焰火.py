# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def numColor(self, root: TreeNode) -> int:
        S = set()

        def read(node):
            if node is None:
                return
            S.add(node.val)
            read(node.left)
            read(node.right)

        read(root)
        return len(S)

tree = TreeNode(3)
# tree.left = TreeNode(3)
# tree.right = TreeNode(3)
print(Solution().numColor(tree))