# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        def search(node):
            if node is None:
                return [[]]

            left = search(node.left)
            right = search(node.right)
            if left == [[]] and right == [[]]:
                return [[node.val]]
            if left == [[]]:
                rs = []
                for r in right:
                    rs.append([node.val] + r)
                return rs
            if right == [[]]:
                rs = []
                for l in left:
                    rs.append([node.val] + l)
                return rs
            rs = []
            # print(node.val, left, right)
            for l in left + right:
                rs.append([node.val] + l)
            return rs

        val = 0
        for path in search(root):
            val += int(''.join(map(str, path)), 2)
        return val