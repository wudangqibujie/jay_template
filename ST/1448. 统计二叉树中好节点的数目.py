# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        rslt = []
        def search(node, val):
            if node is None:
                return
            if node.val >= val:
                rslt.append(node)
            search(node.left, max(val, node.val))
            search(node.right, max(val, node.val))

        search(root, root.val)
        return len(rslt)


s = Solution()
tree = TreeNode(3)
tree.left = TreeNode(1)
tree.left.left = TreeNode(3)
tree.right = TreeNode(4)
tree.right.left = TreeNode(1)
tree.right.right = TreeNode(5)


tree = TreeNode(3)
# tree.left = TreeNode(3)
# tree.left.left = TreeNode(4)
# tree.left.right = TreeNode(2)


print(s.goodNodes(tree))