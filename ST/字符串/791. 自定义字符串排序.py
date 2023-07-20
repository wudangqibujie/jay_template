class Solution:
    def customSortString(self, order: str, s: str) -> str:
        ids = []
        for ix, i in enumerate(s):
            if i not in order:
                ids.append([ix, float('inf')])
            else:
                ids.append([ix, order.index(i)])
        ids = [i[0] for i in sorted(ids, key=lambda x: x[1])]
        return ''.join([s[i] for i in ids])

so = Solution()
order = "cba"
s = "abcd"
order = "a"
s = "c"
print(so.customSortString(order, s))