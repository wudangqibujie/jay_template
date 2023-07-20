from typing import List


class Solution:
    def grayCode(self, n: int) -> List[int]:
        rs = ['0', '1']
        if n == 1:
            return [0, 1]
        for _ in range(n - 1):
            tmp_rs = []
            for r in rs:
                tmp_rs.append(f'0{r}')
            for r in rs[::-1]:
                tmp_rs.append(f'1{r}')
            rs = tmp_rs
        # print(rs)
        return [int(r, 2) for r in rs]



s = Solution()
n = 4
print(s.grayCode(n))
