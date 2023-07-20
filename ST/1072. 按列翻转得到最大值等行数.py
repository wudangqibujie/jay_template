from typing import List


class Solution:
    def maxEqualRowsAfterFlips(self, matrix: List[List[int]]) -> int:
        log_ = dict()
        cnt = 0
        for i in matrix:
            sig = "".join(map(str, i))
            if sig.count('0') == len(sig) or sig.count('1') == len(sig):
                cnt = 1
            if sig in log_:
                log_[sig] += 1
            else:
                log_[sig] = 1
        print(log_)
        for k, v in log_.items():
            reverse_sig = k.replace("0", "2").replace("1", "0").replace("2", "1")
            # print(k, reverse_sig)
            if reverse_sig in log_:
                cnt = max(cnt, v + log_[reverse_sig])
        return cnt


s = Solution()
matrix = [[1,0,0,0,1,1,1,0,1,1,1],
          [1,0,0,0,1,0,0,0,1,0,0],
          [1,0,0,0,1,1,1,0,1,1,1],
          [1,0,0,0,1,0,0,0,1,0,0],
          [1,1,1,0,1,1,1,0,1,1,1]]
print(s.maxEqualRowsAfterFlips(matrix))