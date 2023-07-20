from typing import Optional
from ST.utils import list_to_node



# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return
        lst = None
        node = head
        while node:
            tmp = node.next
            # print(node.val)
            node.next = lst
            lst = node
            node = tmp
        return lst


s = Solution()
head = list_to_node([1,2,])
rs = s.reverseList(head)

while rs:
    print(rs.val)
    rs = rs.next

