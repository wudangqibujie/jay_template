from typing import List


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        rslt = []
        log = dict()
        for st in strs:
            key = ''.join(sorted(st))
            if key in log:
                log[key].append(st)
            else:
                log[key] = [st]
        for k, v in log.items():
            rslt.append(v)
        return rslt


s = Solution()
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
strs = [""]
print(s.groupAnagrams(strs))