class Solution:
    def countAndSay(self, n: int) -> str:


        def describe(num):
            st, ed = 0, 1
            rs = ''
            while st < len(num):
                cnt = 1
                while ed < len(num) and num[st] == num[ed]:
                    ed += 1
                    cnt += 1
                # print(cnt, num[st], ed)
                rs += f'{cnt}{num[st]}'
                st = ed
                ed += 1
            return rs
        nums = '1'
        for _ in range(n - 1):
            nums = describe(nums)
        return nums



s = Solution()
n = 4
print(s.countAndSay(n))