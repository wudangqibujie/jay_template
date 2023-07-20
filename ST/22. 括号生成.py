from typing import List


class Solution:
    def ABC(self, n: int) -> List[str]:
        rslt = []

        def find(s, cnt):
            if s.count(')') > s.count('('):
                return
            if cnt == n * 2:
                if s.count(')') == s.count('('):
                    rslt.append(s)
                return
            find(s + ')', cnt + 1)
            find(s + '(', cnt + 1)

        find('(', 1)
        find(')', 1)
        return rslt

rslt = Solution().ABC(5)
print(rslt)


