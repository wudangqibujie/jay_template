class Trie:

        def __init__(self):
            self._trie = dict()

        def insert(self, word: str) -> None:
            def helper(ix, tmp_dict):
                if ix == len(word):
                    tmp_dict['end'] = True
                    return
                if word[ix] not in tmp_dict:
                    tmp_dict[word[ix]] = dict()
                helper(ix + 1, tmp_dict[word[ix]])
            helper(0, self._trie)

        def search(self, word: str) -> bool:
            def helper(ix, tmp_dict):
                if ix == len(word):
                    return 'end' in tmp_dict
                if word[ix] not in tmp_dict:
                    return False
                return helper(ix + 1, tmp_dict[word[ix]])
            return helper(0, self._trie)
        def startsWith(self, prefix: str) -> bool:
            def helper(ix, tmp_dict):
                if ix == len(prefix):
                    return True
                if prefix[ix] not in tmp_dict:
                    return False
                return helper(ix + 1, tmp_dict[prefix[ix]])

            return helper(0, self._trie)


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)

trie = Trie()
trie.insert("apple")
print(trie.search("apple"))
print(trie.search("app"))
print(trie.startsWith("app"))
trie.insert("app")
print(trie.search("app"))