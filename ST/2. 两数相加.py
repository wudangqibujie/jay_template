from ST.utils import list_to_node
from typing import Optional
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def ABC(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        head = ListNode(-1)
        tmp = head
        add = 0
        while l1 or l2:
            l1_val = l1.val if l1 else 0
            l2_val = l2.val if l2 else 0
            val = l1_val + l2_val + add
            add = val // 10
            bit_vl = val % 10
            tmp.next = ListNode(bit_vl)

            tmp = tmp.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        if add:
            tmp.next = ListNode(add)
        return head.next



s = Solution()
node1 = list_to_node([9, 9, 9])
node2 = list_to_node([0])
rslt = s.ABC(node1, node2)
while rslt:
    print(rslt.val)
    rslt = rslt.next