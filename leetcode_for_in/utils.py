class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class ListToNode:
    def __init__(self, nums):
        self.node = self.list_to_node(nums)

    def __str__(self):
        node = self.node
        rslt = []
        while node:
            rslt.append(node.val)
            node = node.next
        return str(rslt)

    def list_to_node(self, nums):
        if not nums:
            return None
        head = ListNode(nums[0])
        p = head
        for i in nums[1:]:
            p.next = ListNode(i)
            p = p.next
        return head