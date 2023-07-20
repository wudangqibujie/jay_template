class Solution:
    def getKth(self, lo: int, hi: int, k: int) -> int:
        map_ = dict()

        def search(num):
            if num == 1:
                return 0
            if num % 2 == 0:
                new_num = num // 2
            else:
                new_num = 3 * num + 1
            if new_num in map_:
                rs = map_[new_num] + 1
                return rs
            rs = search(new_num) + 1
            map_[new_num] = rs
            return rs
        tmp = dict()
        for num in range(lo, hi + 1):
            map_ = dict()
            tmp[num] = search(num)
        tmp = sorted(tmp.items(), key=lambda x: x[1])
        return tmp[k - 1][0]

s = Solution()
lo = 7
hi = 8
k = 2
print(s.getKth(lo, hi, k))

