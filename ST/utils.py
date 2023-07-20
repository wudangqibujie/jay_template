class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next



def list_to_node(lst):
    head = ListNode(lst[0])
    cur = head
    for i in lst[1:]:
        cur.next = ListNode(i)
        cur = cur.next
    return head


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def list_to_tree(lst):
    if not lst:
        return None
    root = TreeNode(lst[0])
    queue = [root]
    i = 1
    while i < len(lst):
        node = queue.pop(0)
        if lst[i]:
            node.left = TreeNode(lst[i])
            queue.append(node.left)
        i += 1
        if i < len(lst) and lst[i]:
            node.right = TreeNode(lst[i])
            queue.append(node.right)
        i += 1
    return root


def tree_to_list(tree):
    if not tree:
        return []
    queue = [tree]
    res = []
    while queue:
        node = queue.pop(0)
        if node:
            res.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
        else:
            res.append(None)
    while res and res[-1] is None:
        res.pop()
    return res