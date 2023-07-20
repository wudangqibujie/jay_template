# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not tree:
            return 0
        def helper(node):
            if node is None:
                return 0
            l = helper(node.left)
            r = helper(node.right)
            return max(l ,r) + 1

        return helper(root)


tree = TreeNode(1)
# tree.left = TreeNode(2)
# tree.right = TreeNode(3)
# tree.left.left = TreeNode(4)

s = Solution()
print(s.maxDepth(tree))