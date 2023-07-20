# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from typing import Optional
from  queue import Queue
class Solution:
    def replaceValueInTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        que = Queue()
        depth_log = []
        root.partent_sum = root.val
        que.put(root)
        while que.qsize():
            buff_que = Queue()
            val = 0
            while que.qsize():
                pop_node = que.get()
                val += pop_node.val
                if pop_node.left:
                    l_val = pop_node.left.val
                else:
                    l_val = 0
                if pop_node.right:
                    r_val = pop_node.right.val
                else:
                    r_val = 0
                partent_sum = l_val + r_val
                if pop_node.left:
                    pop_node.left.partent_sum = partent_sum
                    buff_que.put(pop_node.left)
                if pop_node.right:
                    pop_node.right.partent_sum = partent_sum
                    buff_que.put(pop_node.right)


                pop_node.depth = len(depth_log)
            depth_log.append(val)
            que = buff_que
        # print(depth_log)
        def search(node):
            if node is None:
                return
            # print(node.val, node.depth, node.partent_sum)
            node.val = depth_log[node.depth] - node.partent_sum

            search(node.left)
            search(node.right)
        search(root)
        return root

s = Solution()
tree = TreeNode(5)
# tree.left = TreeNode(2)
# tree.right = TreeNode(1)
# tree.left.left = TreeNode(1)
# tree.left.right = TreeNode(10)
# tree.right.right = TreeNode(7)


new_tree = s.replaceValueInTree(tree)
print(new_tree.val)
# print(new_tree.left.val)
# print(new_tree.right.val)
# print(new_tree.left.left.val)
# print(new_tree.left.right.val)
# print(new_tree.right.right.val)



