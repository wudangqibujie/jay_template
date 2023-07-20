from copy import copy
class Solution:
    def numSplits(self, s: str) -> int:
        front, tmp_fromt = [], ['0' for _ in range(26)]
        for c in s:
            tmp_fromt[ord(c) - 97] = '1'
            front.append(''.join(tmp_fromt))
            tmp_fromt = copy(tmp_fromt)
        rear, tmp_rear = [], ['0' for _ in range(26)]
        ix = len(s) - 1
        while ix >= 0:
            tmp_rear[ord(s[ix]) - 97] = '1'
            rear.append(''.join(tmp_rear))
            tmp_rear = copy(tmp_rear)
            ix -= 1
        rear.reverse()
        cnt = 0
        for ix in range(len(s) - 1):
            if front[ix].count('1') == rear[ix + 1].count('1'):
                cnt += 1
        return cnt


s = Solution()
ss = "a" * 100000
print(s.numSplits(ss))