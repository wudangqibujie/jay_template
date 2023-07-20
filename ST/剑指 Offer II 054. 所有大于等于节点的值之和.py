# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def convertBST(self, root: TreeNode) -> TreeNode:

        stack = [(root, False)]
        lst_val = 0
        while stack:
            pop_node, pop_status = stack.pop()
            if pop_node is None:
                continue
            if pop_status:
                pop_node.val += lst_val
                lst_val = pop_node.val
            else:
                stack.append((pop_node.left, False))
                stack.append((pop_node, True))
                stack.append((pop_node.right, False))

        return root

from ST.utils import list_to_tree
import json
tree = list_to_tree(json.loads('[4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]'))


tree = TreeNode(4)
tree.left = TreeNode(1)
tree.right = TreeNode(6)
tree.left.left = TreeNode(0)
tree.left.right = TreeNode(2)

node = Solution().convertBST(tree)
print(node.val)
print(node.left.val)
print(node.right.val)
print(node.left.left.val)
print(node.left.right.val)
