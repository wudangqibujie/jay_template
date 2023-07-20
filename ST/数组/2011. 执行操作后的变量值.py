from typing import List


class Solution:
    def finalValueAfterOperations(self, operations: List[str]) -> int:
        rs = 0
        for op in operations:
            if op in ['--X', 'X--']:
                rs -= 1
            else:
                rs += 1
        return rs


s = Solution()
operations = ["--X","X++","X++"]
operations = ["++X","++X","X++"]
operations = ["X++","++X","--X","X--"]
print(s.finalValueAfterOperations(operations))