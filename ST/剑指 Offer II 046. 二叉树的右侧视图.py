# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from typing import List
from queue import Queue
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        rslt = []
        que = Queue()
        que.put(root)
        while que.qsize():
            buff_que = Queue()
            rs = None
            while que.qsize():
                pop_node = que.get()
                if rs is None:
                    rs = pop_node.val
                    rslt.append(rs)
                if pop_node.right:
                    buff_que.put(pop_node.right)
                if pop_node.left:
                    buff_que.put(pop_node.left)
            que = buff_que
        return rslt


s = Solution()
tree = TreeNode(1)
tree.left = TreeNode(2)
tree.right = TreeNode(3)
tree.left.right = TreeNode(5)
tree.right.right = TreeNode(4)


tree = TreeNode(1)
tree.right = TreeNode(3)
tree = None
print(s.rightSideView(tree))


