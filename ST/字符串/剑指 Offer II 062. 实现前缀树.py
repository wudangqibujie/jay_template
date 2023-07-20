class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.d = dict()
        self.inserted = None

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        self.inserted = word
        tmp_d = self.d
        for c in word:
            if c not in tmp_d:
                tmp_d[c] = dict()
            tmp_d = tmp_d[c]
        tmp_d['WORD'] = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        tmp_d = self.d
        for c in word:
            if c not in tmp_d:
                return False
            tmp_d = tmp_d[c]
        # print(tmp_d)
        if 'WORD' not in tmp_d:
            return False
        return True

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        if self.inserted is None:
            return False
        for ix, c in enumerate(prefix):

            if ix >= len(self.inserted) or c != self.inserted[ix]:
                return False
        return True



# Your Trie object will be instantiated and called as such:
obj = Trie()
# obj.insert('apple')
# param_2 = obj.search('apple')
# print(obj.search('app'))
param_3 = obj.startsWith('app')
# obj.insert('app')
# print(param_2, param_3)
# print(obj.search('app'))