from typing import List


class Solution:
    def constructArr(self, a: List[int]) -> List[int]:
        val = 1
        zero = 0
        non_zero_val = 1
        for i in a:
            if i == 0:
                zero += 1
            val *= i
            if i != 0:
                non_zero_val *= i
        if zero > 1:
            return [0 for _ in range(len(a))]
        rslt = []
        # print(val)
        for i in a:
            if i == 0:
                rslt.append(non_zero_val)
            else:
                rslt.append(val // i)
        return rslt


s = Solution()
a = [2]
print(s.constructArr(a))

