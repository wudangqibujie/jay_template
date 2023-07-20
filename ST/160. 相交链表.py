# Definition for singly-linked list.
from typing import Optional
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        len1, len2 = 0, 0
        tmpA, tmpB = headA, headB
        while tmpA:
            len1 += 1
            tmpA = tmpA.next
        while tmpB:
            len2 += 1
            tmpB = tmpB.next

        if len1 > len2:
            node_long, node_short = headA, headB
        else:
            node_long, node_short = headB, headA
        flg = 1
        while node_long and node_long != node_short:
            node_long = node_long.next
            if flg <= abs(len1 - len2):
                flg += 1
                continue
            node_short = node_short.next
        return node_long


node1 = ListNode(4)
node1.next = ListNode(5)


node_l = ListNode(1)
node_l.next = ListNode(2)
node_l.next.next = node1

node_s = ListNode(-1)
node_s.next = ListNode(0)
node_s.next.next = ListNode(1)
node_s.next.next.next = node1

print(node_l.next.next == node_s.next.next.next)

s = Solution()
rs = s.getIntersectionNode(node_l, node_s)
print(rs.val)