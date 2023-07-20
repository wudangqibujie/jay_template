from typing import List

class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        rslt = []

        def search(target, candi, route):
            if len(route) == k:
                if target == 0:
                    route.sort()
                    if route not in rslt:
                        rslt.append(route)
                return

            for c in candi:
                new_target = target - c
                new_candi = [i for i in candi if i != c]
                search(new_target, new_candi, route + [c])
        search(n, [i for i in range(1, 10)], [])
        return rslt


s = Solution()
k = 7
n = 7

k = 3
n = 9
k = 4
n = 1


k = 9
n = 45
print(s.combinationSum3(k, n))