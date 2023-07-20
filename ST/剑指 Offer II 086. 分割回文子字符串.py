from typing import List

class Solution:
    def partition(self, s: str) -> List[List[str]]:
        map_idx = [[] for _ in range(len(s))]
        tmp = []
        for ix in range(len(s)):
            l_ix, r_ix = ix, ix
            while l_ix >= 0 and r_ix <= len(s) - 1 and s[l_ix] == s[r_ix]:
                tmp.append((l_ix, r_ix + 1))
                l_ix -= 1
                r_ix += 1
            l_ix, r_ix = ix - 1, ix
            while l_ix >= 0 and r_ix <= len(s) - 1 and s[l_ix] == s[r_ix]:
                tmp.append((l_ix, r_ix + 1))
                l_ix -= 1
                r_ix += 1
        print(tmp)
        for ix in range(len(tmp)):
            map_idx[tmp[ix][0]].append(tmp[ix])
        print(map_idx)

s = Solution()
S = 'googelle'
print(s.partition(S))