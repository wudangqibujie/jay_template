class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        rs = s[0]

        def check(word, ch):
            ix = word.index(ch)
            nx_ix = ix + 1
            if nx_ix == len(word):
                return False
            return ord(word[ix]) > ord(word[nx_ix])

        for ix in range(1, len(s)):
            if s[ix] not in rs:
                rs += s[ix]
            else:
                if check(rs, s[ix]):
                    rs = rs.replace(s[ix], '')
                    rs += s[ix]
            print(rs, s[ix])
        print("--------")
        return rs


s = Solution()
st = "bcabc"
print(s.removeDuplicateLetters(st))