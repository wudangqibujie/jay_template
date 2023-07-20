from typing import List


class WordsFrequency:

    def __init__(self, book: List[str]):
        self.vocab = self._init_vocab(book)

    def _init_vocab(self, book):
        vocab = {}
        for word in book:
            if word not in self.vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
        return vocab

    def get(self, word: str) -> int:
        if word not in self.vocab:
            return 0
        else:
            return self.vocab[word]



# Your WordsFrequency object will be instantiated and called as such:
# obj = WordsFrequency(book)
# param_1 = obj.get(word)