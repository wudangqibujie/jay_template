# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.stack = [(root, False)]

    def next(self) -> int:
        if self.hasNext():
            pop_node, node_status = self.stack.pop()
            if node_status:
                return pop_node.val
            else:
                if pop_node.right:
                    self.stack.append((pop_node.right, False))
                self.stack.append((pop_node, True))
                if pop_node.left:
                    self.stack.append((pop_node.left, False))
                return self.next()

    def hasNext(self) -> bool:
        return len(self.stack) > 0


# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()

tree = TreeNode(7)
tree.left = TreeNode(3)
tree.right = TreeNode(15)
tree.right.left = TreeNode(9)
tree.right.right = TreeNode(20)

obj = BSTIterator(tree)

print(obj.next())
print(obj.next())
print(obj.hasNext())
print(obj.next())
print(obj.hasNext())
print(obj.next())
print(obj.hasNext())
print(obj.next())
print(obj.hasNext())
print(obj.next())

