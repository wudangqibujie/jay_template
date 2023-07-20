# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
from typing import List
from queue import Queue
class Solution:
    def listOfDepth(self, tree: TreeNode) -> List[ListNode]:
        que = Queue()
        que.put(tree)
        rs = []
        while que.qsize():
            buff_que = Queue()
            tmp = ListNode(-1)
            head = tmp
            while que.qsize():
                node = que.get()
                head.next = ListNode(node.val)
                head = head.next
                if node.left:
                    buff_que.put(node.left)
                if node.right:
                    buff_que.put(node.right)
            rs.append(tmp.next)
            que = buff_que
        return rs
s = Solution()
tree = TreeNode(1)
# tree.left = TreeNode(2)
# tree.right = TreeNode(3)
# tree.left.left = TreeNode(4)
# tree.left.right = TreeNode(5)
# tree.left.left.left = TreeNode(8)
# tree.right.right = TreeNode(7)

rs = s.listOfDepth(tree)

for n in rs:
    print('0000000000000000000')
    while n:
        print(n.val)
        n = n.next