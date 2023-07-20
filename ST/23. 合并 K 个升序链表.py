from typing import List, Optional
from ST.utils import list_to_node

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def ABC(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:

        if not lists:
            return None
        if len(lists) == 1:
            return lists[0]

        new = self.mergeTwoLists(lists[0], lists[1])
        head = new
        for i in lists[2:]:
            new = self.mergeTwoLists(new, i)
            head = new
        return head


    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        head = ListNode(-1)
        l = head
        while list1 or list2:
            if list1 is None:
                l.next = list2
                list2 = list2.next
                l = l.next
                continue
            elif list2 is None:
                l.next = list1
                list1 = list1.next
                l = l.next
                continue
            if list1.val < list2.val:
                l.next = list1
                list1 = list1.next
            else:
                l.next = list2
                list2 = list2.next
            l = l.next
        return head.next


s = Solution()

lists = [[1,4,5],[1,3,4],[2,6]]
nums = [
    list_to_node(i) for i in lists
]

nums = [None, list_to_node([1, 2])]

rslt = s.ABC(nums)
while rslt:
    print(rslt.val)
    rslt = rslt.next