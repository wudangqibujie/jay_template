from typing import List


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        rslt = []
        readed = []
        def search(now, now_target):
            if now_target == 0:
                if now not in rslt:
                    rslt.append(now)
                return
            for c in candidates:
                new_now = now + [c]
                new_now.sort()
                if new_now in readed or now_target - c < 0:
                    continue
                search(new_now, now_target - c)
                readed.append(new_now)

        search([], target)
        return rslt


s = Solution()
candidates = [2,3,6,7]
target = 7
candidates = [2,3,5]
target = 8
candidates = [2]
target = 1
print(s.combinationSum(candidates, target))