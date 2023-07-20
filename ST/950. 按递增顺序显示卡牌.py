from typing import List

class Solution:
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        mock_ids = [i for i in range(len(deck))]
        ids = []
        while mock_ids:
            ids.append(mock_ids[0])
            if len(mock_ids) == 1:
                break
            tmp = mock_ids[1]
            mock_ids = mock_ids[2:] + [tmp]
        # print(ids)
        rs = [None for _ in range(len(deck))]
        deck.sort()
        # print(deck)
        for ix in range(len(deck)):
            rs[ids[ix]] = deck[ix]
        return rs

s = Solution()
deck = [17]
print(s.deckRevealedIncreasing(deck))