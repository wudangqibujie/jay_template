# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from typing import Optional
class Solution:
    def reverseOddLevels(self, root: Optional[TreeNode]) -> Optional[TreeNode]:

        container = [root]
        depth = 0
        while container:
            tmp_container = []
            if depth % 2 == 1:
                l_ix, r_ix = 0, len(container) - 1
                while l_ix < r_ix:
                    l_node = container[l_ix]
                    r_node = container[r_ix]
                    l_node.left, l_node.right = r_node.left, r_node.right
                    l_ix += 1
                    r_ix -= 1
            while container:
                pop_node = container.pop()
                if pop_node.left:
                    tmp_container.append(pop_node.left)
                if pop_node.right:
                    tmp_container.append(pop_node.right)
                depth += 1
            container = tmp_container
        return root


s = Solution()
tree = TreeNode(2)
tree.left = TreeNode(3)
tree.right = TreeNode(5)
tree.left.left = TreeNode(8)
tree.left.right = TreeNode(13)
tree.right.right = TreeNode(34)
tree.right.left = TreeNode(21)
rs = s.reverseOddLevels(tree)
print(rs.val)
print(rs.left.val)
