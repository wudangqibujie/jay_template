# Definition for singly-linked list.
from typing import Optional
from ST.utils import list_to_node
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeNodes(self, head: Optional[ListNode]) -> Optional[ListNode]:

        node_0 = head
        node_nxt = head.next
        lst_node_0 = None
        while node_nxt:
            if node_nxt.val == 0:
                if node_nxt.next is None:
                    node_0.next = None
                    break
                # print('node_0: ', node_0.val)
                node_0.next = node_nxt
                node_0 = node_nxt
            else:
                node_0.val += node_nxt.val
            node_nxt = node_nxt.next
        # print(node_0.val)
        return head


node = list_to_node([0,1,1, 0, 1,0])

rs = Solution().mergeNodes(node)
while rs:
    print(rs.val)
    rs = rs.next