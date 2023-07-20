from typing import List
from copy import copy

class Solution:
    def distributeCookies(self, cookies: List[int], k: int) -> int:
        if len(cookies) == k:
            return max(cookies)
        cookies.sort()
        init_nums = [cookies.pop() for _ in range(k)]
        cookies.reverse()
        dp = []
        tmp = []
        for ix in range(k):
            buff = copy(init_nums)
            buff[ix] += cookies[0]
            tmp.append(buff)
        dp.append(tmp)
        for num in cookies[1: ]:
            tmp = []
            for ix in range(k):
                buff = copy(dp[-1])
                min_idx = buff[ix].index(min(buff[ix]))
                # print(buff[ix][min_idx])
                buff[ix][min_idx] += num
                # print(buff)
                tmp.append(buff[ix])
            dp.append(tmp)
        # for i in dp:
        #     print(i)
        return min([max(i) for i in dp[-1]])

s = Solution()
cookies = [1,2,3,4,5,6,7,8,9,10]
cookies = [8,15,10,20,8, 6]
k = 4
# cookies = [6,1, ]
# k = 2
print(s.distributeCookies(cookies, k))