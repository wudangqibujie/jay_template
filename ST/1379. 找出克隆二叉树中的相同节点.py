# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        if not original:
            return

        def read(node1, node2):
            if node1 is None:
                return
            if node1.val == target:
                return node2
            left_rs = read(node1.left, node2.left)
            right_rs = read(node1.right, node2.right)
            if left_rs:
                return left_rs
            if right_rs:
                return right_rs

        return read(original, cloned)


s = Solution()
tree1 = TreeNode(7)
# tree1.left = TreeNode(4)
# tree1.right = TreeNode(3)
# tree1.right.left = TreeNode(6)
# tree1.right.right = TreeNode(19)

tree2 = TreeNode(7)
# tree2.left = TreeNode(4)
# tree2.right = TreeNode(3)
# tree2.right.left = TreeNode(6)
# tree2.right.right = TreeNode(19)


rs = s.getTargetCopy(tree1, tree2, 7)
print(rs.val)

