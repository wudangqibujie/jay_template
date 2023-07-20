# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
from typing import List
import heapq

class Solution:
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        rs = []
        heapq.heapify(rs)
        def read(node):
            if node is None:
                return
            read(node.left)
            heapq.heappush(rs, node.val)
            read(node.right)

        read(root1)
        read(root2)
        rs.sort()
        return rs


tree1 = TreeNode(2)
tree1.left = TreeNode(1)
tree1.right = TreeNode(4)

tree2 = TreeNode(1)
tree2.left = TreeNode(0)
tree2.right = TreeNode(3)

s = Solution()
print(s.getAllElements(tree1, tree2))



















