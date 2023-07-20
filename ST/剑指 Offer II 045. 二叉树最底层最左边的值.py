# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
from queue import Queue
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
        que = Queue()
        que.put(root)
        lst = None
        while que.qsize():
            buff_que = Queue()
            while que.qsize():
                node = que.get()
                lst = node.val
                # print(node.val)
                if node.right:
                    buff_que.put(node.right)
                if node.left:
                    buff_que.put(node.left)
            que = buff_que
        return lst



tree = TreeNode(1)
# tree.left = TreeNode(2)
# tree.left.left = TreeNode(4)
# tree.right = TreeNode(3)
# tree.right.left = TreeNode(5)
# tree.right.left.left = TreeNode(7)
# tree.right.right = TreeNode(6)


s = Solution()
print(s.findBottomLeftValue(tree))