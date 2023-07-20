from typing import  List

class Solution:
    def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
        def match(word, p):
            l = dict()
            for ix in range(len(word)):
                if p[ix] not in l:
                    l[p[ix]] = word[ix]
                else:
                    if l[p[ix]] != word[ix]:
                        return False
            return True

        rs = []
        for word in words:
            if match(word, pattern) and match(pattern, word):
                rs.append(word)
        return rs


s = Solution()
words = ["abc","deq","mee","aqq","abb","ccc"]
pattern = "aaa"
print(s.findAndReplacePattern(words, pattern))