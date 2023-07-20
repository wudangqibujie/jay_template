from typing import List

class Solution:
    def maxScoreWords(self, words: List[str], letters: List[str], score: List[int]) -> int:
        score_map = dict()
        for ix, s in enumerate(score):
            score_map[chr(ord('a') + ix)] = s
        word_score = []
        print(score_map)
        for word in words:
            word_score.append(sum([score_map[c] for c in word]))
        word_score = sorted(zip(word_score, words), key=lambda x: x[0], reverse=True)
        print(word_score)
        word_strip = dict()
        for l in letters:
            if l not in word_strip:
                word_strip[l] = 1
            else:
                word_strip[l] += 1
        print(word_strip)



words = ["dog","cat","dad","good"]
letters = ["a","a","c","d","d","d","g","o","o"]
score = [1,0,9,5,0,0,3,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0]
s = Solution()
s.maxScoreWords(words, letters, score)
