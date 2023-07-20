# Definition for a binary tree node.
from typing import Optional
from ST.utils import list_to_tree, tree_to_list
from queue import Queue
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def deepestLeavesSum(self, root: Optional[TreeNode]) -> int:
        que = Queue()
        rs = None
        que.put(root)
        while True:
            if que.qsize() == 0:
                return rs
            buff_que = Queue()
            rs = 0
            while que.qsize():
                node = que.get()
                rs += node.val
                if node.left:
                    buff_que.put(node.left)
                if node.right:
                    buff_que.put(node.right)
            que = buff_que



import json
d = json.loads('[6]')
tree = list_to_tree(d)

s = Solution()
print(s.deepestLeavesSum(tree))
