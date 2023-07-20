# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def sumEvenGrandparent(self, root: TreeNode) -> int:
        rslt = 0
        def helper(node):
            nonlocal rslt
            if node is None:
                return 0
            rs = 0
            if node.left:
                if node.left.left:
                    rs += node.left.left.val
                if node.left.right:
                    rs += node.left.right.val
            if node.right:
                if node.right.left:
                    rs += node.right.left.val
                if node.right.right:
                    rs += node.right.right.val
            # print(node.val, rs)
            helper(node.left)
            helper(node.right)
            rslt += rs if node.val % 2 == 0 else 0

        helper(root)
        return rslt


from ST.utils import list_to_tree
import json
tree = list_to_tree(json.loads('[2, 2, 3, 4]'))
s = Solution()
print(s.sumEvenGrandparent(tree))



