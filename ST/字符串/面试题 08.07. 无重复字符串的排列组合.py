from typing import List


class Solution:
    def permutation(self, S: str) -> List[str]:
        nums = list(S)
        readed = set()
        rslt = []
        def search(now, candi):
            # print(now, candi)
            if not candi:
                if now not in nums:
                    rslt.append(now)
                return
            for c in candi:
                new_can = [i for i in candi if i != c]
                if now + c in readed:
                    continue
                search(now + c, new_can)
                readed.add(now + c)

        search('', nums)
        return rslt


s = Solution()
S = "ab"
print(s.permutation(S))