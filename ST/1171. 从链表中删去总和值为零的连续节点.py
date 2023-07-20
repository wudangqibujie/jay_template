from typing import Optional
from ST.utils import list_to_node

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def ABC(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head.next is None and head.val == 0:
            return None
        def process(head):

            tmp = head
            log = dict()
            log_trie = dict()
            # log_trie[0] = [0]
            lst = None
            trie_sum = [0]
            ix = 0
            while head:
                log[ix] = {
                    'now': head,
                    'lst': lst
                }
                trie_sum.append(trie_sum[-1] + head.val)
                if trie_sum[-1] in log_trie:
                    log_trie[trie_sum[-1]].append(ix)
                else:
                    log_trie[trie_sum[-1]] = [ix]
                lst = head
                head = head.next
                ix += 1
            max_len = -1
            info = None
            print(log)
            print(log_trie)
            for k, v in log_trie.items():
                if k == 0:
                    if max(v) > max_len:
                        info = k, max(v), -1
                    continue
                if len(v) < 2:
                    continue
                if max(v) - min(v) > max_len:
                    info = k, max(v), min(v)
            if info is None:
                return ''
            # print(info)
            _, front, frst = info
            if frst == -1:
                return log[front]['now'].next
            else:
            # print(frst, front)
                log[frst]['now'].next = log[front]['now'].next
            return tmp

        lst = head
        while True:
            if lst is None:
                return lst
            now = process(lst)
            if isinstance(now, str):
                return lst
            lst = now



nums = [1,2,-3,3,1]
# nums = [1,2,3,-3,4]
nums = [1,2,3,-3,-2, 1, 4, 1, 1, -2, -4]
nums = [0, 1, -1]
head = list_to_node(nums)
rslt = Solution().ABC(head)

while rslt:
    print(rslt.val)
    rslt = rslt.next