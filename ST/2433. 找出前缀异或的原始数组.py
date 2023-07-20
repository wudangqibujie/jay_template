from typing import List


class Solution:
    def findArray(self, pref: List[int]) -> List[int]:
        ans = [pref[0]]
        for i in range(1, len(pref)):
            ans.append(pref[i] ^ pref[i - 1])
        return ans



s = Solution()
pref = [5,2,0,3,1]
print(s.findArray(pref))