from typing import Optional
from utils import ListToNode


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        rslt_node = ListNode(-1)
        added_node = rslt_node
        additional_val = 0
        while l1 or l2:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            added_node.next = ListNode((v1 + v2 + additional_val) % 10)
            additional_val = (v1 + v2 + additional_val) // 10
            added_node = added_node.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        if additional_val != 0:
            added_node.next = ListNode(additional_val)
        return rslt_node.next



if __name__ == '__main__':
    l1 = ListToNode([9, 9, 9, 9])
    l2 = ListToNode([1])

    rslt = Solution().addTwoNumbers(l1.node, l2.node)
    while rslt:
        print(rslt.val)
        rslt = rslt.next