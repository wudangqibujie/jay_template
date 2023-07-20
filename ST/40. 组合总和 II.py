from typing import List


class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        rslt = []
        readed = set()

        def search(target, candi, route):
            candi.sort()
            if f'{target}:{candi}' in readed:
                return
            if target == 0:
                rslt.append(route)
                return
            if target < 0:
                return
            for ix, c in enumerate(candi):
                new_target = target - c
                new_candi = [j for jx, j in enumerate(candi) if jx != ix]
                search(new_target, new_candi, route + [c])
                new_candi.sort()
                readed.add(f'{new_target}:{new_candi}')

        search(target, candidates, [])
        return rslt


s = Solution()
candidates = [10,1,2,7,6,1,5]
candidates.sort()
# print(candidates)
target = 8

candidates = [2,5,2,1,2,2,5,2,1,2,2,5,2,1,2,2,5,2,1,2,2,5,2,1,2,2,5,2,1,2,2,5,2,1,2,2,5,2,1,2,2,5,2,1,2,2,5,2,1,2,2,5,2,1,2]
target = 50
print(s.combinationSum2(candidates, target))