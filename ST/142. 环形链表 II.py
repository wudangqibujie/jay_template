# Definition for singly-linked list.
from typing import Optional
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast, slow = head, head
        flag = 0
        while 1:
            if fast is None:
                return None
            fast = fast.next
            flag += 1
            if flag % 2 == 0:
                slow = slow.next
            if fast == slow and flag % 2 == 0:
                break
        new = head
        rs = 0
        # print(new.val, fast.val ,slow.val)
        while new != fast:
            new = new.next
            fast = fast.next
            rs += 1
        return new


s = Solution()
l1 = ListNode(3)
l1.next = ListNode(2)
l1.next.next = l1


s = Solution()
print(s.detectCycle(l1).val)