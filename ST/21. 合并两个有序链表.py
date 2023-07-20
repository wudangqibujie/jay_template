from ST.utils import list_to_node
from typing import Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        head = ListNode(-1)
        l = head
        while list1 or list2:
            if list1 is None:
                l.next = ListNode(list2.val)
                list2 = list2.next
                l = l.next
                continue
            elif list2 is None:
                l.next = ListNode(list1.val)
                list1 = list1.next
                l = l.next
                continue
            if list1.val < list2.val:
                l.next = ListNode(list1.val)
                list1 = list1.next
            else:
                l.next = ListNode(list2.val)
                list2 = list2.next
            l = l.next
        return head.next


l1 = list_to_node([1, 2, 4])
l2 = list_to_node([1, 3, 4])
l1 = None
l2 = list_to_node([1])
l = Solution().mergeTwoLists(l1, l2)

while l:
    print(l.val)
    l = l.next