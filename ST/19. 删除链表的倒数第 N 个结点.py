from typing import Optional
from ST.utils import list_to_node

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        if not head:
            return
        raer_lst = None
        rear = None
        front = head
        flag = 1
        while front:
            if flag == n:
                rear = head
            elif flag > n:
                raer_lst = rear
                rear = rear.next
            front = front.next
            flag += 1
        if raer_lst is None:
            return head.next
        raer_lst.next = rear.next
        return head


node = list_to_node([1])
n = 4
rslt = Solution().removeNthFromEnd(node, n)

while rslt:
    print(rslt.val)
    rslt = rslt.next