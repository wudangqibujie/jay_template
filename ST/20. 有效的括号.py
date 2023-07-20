class Solution:
    def ABC(self, s: str) -> bool:
        match = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for si in s:
            if si in match:
                stack.append(si)
            else:
                if not stack:
                    return False
                top_s = stack.pop()
                if match[top_s] != si:
                    return False
        return not stack


so = Solution()
s = "(]"
# s = "()[]{}"
s = '('
print(so.ABC(s))