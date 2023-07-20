from typing import List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left: TreeNode = left
        self.right: TreeNode = right

from queue import Queue
class Solution:
    def pathInZigZagTree(self, label: int) -> List[int]:
        def get_range(n):
            return 2 ** (n - 1), 2 ** n - 1
        que = Queue()
        tree = TreeNode(1)
        root = tree
        que.put(root)
        n = 2
        while get_range(n)[0] <= label:
            buff_que = Queue()
            child_min, child_max= get_range(n)
            # print(child_min, child_max)
            children = [TreeNode(n) for n in range(child_min, child_max + 1)]
            if n % 2 != 0:
                children.reverse()
            while que.qsize():
                pop_node = que.get()
                pop_node.left = children.pop()
                pop_node.right = children.pop()
                buff_que.put(pop_node.left)
                buff_que.put(pop_node.right)
            que = buff_que
            n += 1


        def read(node):
            if node is None:
                return []
            if node.val == label:
                return [node.val]
            left_val = read(node.left)
            right_val = read(node.right)
            if left_val:
                return [node.val] + left_val
            if right_val:
                return [node.val] + right_val

        return read(root)




s = Solution()
label = 787046
print(s.pathInZigZagTree(label))