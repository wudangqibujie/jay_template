class Solution:
    def isScramble(self, s1: str, s2: str) -> bool:
        map_ = dict()
        def search(words, sub_word):
            if len(words) == 1:
                return [words]
            tmp = []
            for ix in range(1, len(words)):
                l_word = words[: ix]
                r_word = words[ix:]
                print(f"{sub_word} | {l_word} | {r_word}")
                if sub_word[0] not in l_word or sub_word[-1] not in r_word:  # 有可能是右边的第一个字母
                    continue
                print("bingo")
                l_word_lst = map_.get(l_word, search(l_word, sub_word[: ix]))
                r_word_lst = map_.get(r_word, search(r_word, sub_word[ix:]))
                for l in l_word_lst:
                    for r in r_word_lst:
                        if l + r not in tmp:
                            tmp.append(l + r)
                        if r + l not in tmp:
                            tmp.append(r + l)
            map_[words] = tmp
            return tmp

        lst = search(s1, s2)
        print(lst)
        return s2 in lst


so = Solution()
s1 = "great"
s2 = "rgeat"
print(so.isScramble(s1, s2))
