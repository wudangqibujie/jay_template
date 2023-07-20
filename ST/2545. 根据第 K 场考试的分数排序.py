from typing import List


class Solution:
    def sortTheStudents(self, score: List[List[int]], k: int) -> List[List[int]]:
        idx_log = dict()
        list_k = [score[i][k] for i in range(len(score))]
        for ix, i in enumerate(list_k):
            if i not in idx_log:
                idx_log[i] = [ix]
            else:
                idx_log[i].append(ix)

        list_k_sorted = sorted(list_k, reverse=True)
        readed = set()
        # print(list_k_sorted, idx_log)
        for need_ix, i in enumerate(list_k_sorted):
            if i == list_k[need_ix]:
                continue
            old_idx = idx_log[i].pop()
            # print(old_idx, need_ix)
            if old_idx in readed:
                continue
            for j in range(len(score[0])):
                score[need_ix][j], score[old_idx][j] = score[old_idx][j], score[need_ix][j]
            readed.add(need_ix)
        for s in score:
            print(s)
        return score


s = Solution()
score = [[10,6,9,1],
         [7,5,11,2],
         [7, 5, 11, 2],
         [4,8,3,15]]
k = 0

# score = [[3,4],[5,6]]
# k = 0
s.sortTheStudents(score, k)

