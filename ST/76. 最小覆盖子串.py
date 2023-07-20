class Solution:
    def minWindow(self, s: str, t: str) -> str:
        ix = 0
        while ix < len(s) and s[ix] not in t:
            ix += 1
        i = j = ix
        target_status = dict()
        status = dict()

        index_log = []
        for ix, ii in enumerate(s):
            if ii in t:
                index_log.append(ix)
        # print(index_log)
        for ti in t:
            if ti not in target_status:
                target_status[ti] = 1
            else:
                target_status[ti] += 1
            if ti not in status:
                status[ti] = 0

        if len(t) == 1:
            if t in s:
                return t
        # print(status)
        # print(target_status)
        # print('---')
        rs = ''
        def check():
            for k, v in target_status.items():
                if v > status[k]:
                    return False
            return True

        while j < len(s):
            if s[j] not in t:
                j += 1
                continue

            status[s[j]] += 1
            if check():
                # print(status, i, j, s[i: j + 1])
                if not rs:
                    rs = s[i: j + 1]
                else:
                    if len(rs) > (j - i + 1):
                        rs = s[i: j + 1]
                status[s[i]] -= 1
                i = index_log[index_log.index(i) + 1]
                while check():
                    if not rs:
                        rs = s[i: j + 1]
                    else:
                        if len(rs) > (j - i + 1):
                            rs = s[i: j + 1]
                    status[s[i]] -= 1
                    i = index_log[index_log.index(i) + 1]

            j += 1
        return rs


so = Solution()

# print(len(s))
print(so.minWindow(s, t))