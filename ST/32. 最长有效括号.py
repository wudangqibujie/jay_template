
class Solution:
    def ABC(self, s: str) -> int:
        stack = []
        for ix, char_ in enumerate(s):
            if not stack:
                stack.append((ix, char_))
                continue
            if char_ == '(':
                stack.append((ix, char_))
            else:
                _, top_char = stack[-1]
                if top_char == '(':
                    stack.pop()
                else:
                    stack.append((ix, char_))
        if not stack:
            return len(s)
        # print(stack)
        distance = [-1] + [i[0] for i in stack] + [len(s)]
        rslt = 0
        for ix in range(1, len(distance)):
            rslt = max(rslt, distance[ix] - distance[ix - 1] - 1)
        return rslt

s = Solution()
print(s.ABC("()()())))))))))))))()()())"))
