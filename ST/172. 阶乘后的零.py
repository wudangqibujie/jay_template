class Solution:
    def trailingZeroes(self, n: int) -> int:


        def find(num):
            map_ = dict()
            ix = 1
            while ix * num <= n:
                val = ix * num
                cnt = 0
                while val % num == 0:
                    if val in map_:
                        cnt += map_[val]
                        break
                    cnt += 1
                    val //= num
                map_[ix * num] = cnt
                ix += 1
            # print(map_)
            return sum(map_.values())

        cnt_2 = find(2)
        cnt_5 = find(5)
        return min(cnt_2, cnt_5)


s = Solution()
n = 4
print(s.trailingZeroes(n))

val = 1
for i in range(1, n + 1):
    val *= i
s_val = str(val)
print(len(s_val) - len(s_val.rstrip('0')))
