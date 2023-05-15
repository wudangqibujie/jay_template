from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        log = dict()
        for ix, i in enumerate(nums):
            if i in log:
                log[i].append(ix)
            else:
                log[i] = [ix]

        for ix, i in enumerate(nums):
            if target - i in log:
                if target - i == i:
                    if len(log[i]) > 1:
                        return log[i][:2]
                else:
                    return [ix, log[target - i][0]]


if __name__ == '__main__':
    nums = [2,7,11,15]
    target = 9
    print(Solution().twoSum(nums, target))