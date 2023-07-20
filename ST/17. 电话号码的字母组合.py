from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        dic = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
               '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        rslt = ['']
        for digit in digits:
            buff = []
            for r in rslt:
                buff.extend([r + a for a in dic[digit]])
            rslt = buff
        return rslt

s = Solution()
digits = "2323"
print(s.letterCombinations(digits))