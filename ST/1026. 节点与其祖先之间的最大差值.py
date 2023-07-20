# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from typing import Optional
class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        rs = float('-inf')
        def search(node):
            nonlocal rs
            if node is None:
                return None, None
            left_min, left_max = search(node.left)
            right_min, right_max = search(node.right)

            if left_max is None and right_max is None:
                return [node.val, node.val]
            if left_max is None:
                rs_tmp = max(abs(node.val - right_min), abs(node.val - right_max))
                rs = max(rs, rs_tmp)
                return [min(right_min, node.val), max(right_max, node.val)]
            if right_max is None:
                rs_tmp = max(abs(node.val - left_min), abs(node.val - left_max))
                rs = max(rs, rs_tmp)
                return [min(left_min, node.val), max(left_max, node.val)]

            min_val = min(left_min, right_min)
            max_val = max(left_max, right_max)



            rs_tmp = max(abs(node.val - min_val), abs(node.val - max_val))
            rs = max(rs, rs_tmp)
            # print(node.val, rs)
            return [min(min_val, node.val), max(max_val, node.val)]

        search(root)
        return rs


s = Solution()
tree = TreeNode(1)
tree.right = TreeNode(2)
tree.right.right = TreeNode(0)
tree.right.right.left = TreeNode(3)


tree = TreeNode(0)
tree.left = TreeNode(3)
# tree.left.left = TreeNode(1)
# tree.left.right = TreeNode(6)
# tree.left.right.left = TreeNode(4)
# tree.left.right.right = TreeNode(7)
# tree.right = TreeNode(10)
# tree.right.right = TreeNode(14)
# tree.right.right.left = TreeNode(13)


print(s.maxAncestorDiff(tree))