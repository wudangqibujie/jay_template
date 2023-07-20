from typing import List


class Solution:
    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        def search(g):
            dist_t, gather_t = 0, 0
            gather_t += garbage[0].count(g)
            dist_status = [g in garbage[0]]
            for ix in range(1, len(garbage)):
                gather_t += (garbage[ix].count(g))
                dist_status.append(g in garbage[ix])
            if gather_t == 0:
                return 0
            lst_dis_ix = len(garbage) - 1
            while lst_dis_ix >= 0:
                if dist_status[lst_dis_ix]:
                    break
                lst_dis_ix -= 1
            st = 0
            while st < lst_dis_ix:
                dist_t += travel[st]
                st += 1
            return gather_t + dist_t
        # print(search('M'))
        # print(search('G'))
        # print(search('P'))
        return search('M') + search('G') + search('P')


s = Solution()
garbage = ["G","P","GP","GG"]
travel = [2,4,3]
garbage = ["MMM"]
travel = [3]
print(s.garbageCollection(garbage, travel))