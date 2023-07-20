from typing import List

class Solution:
    def wateringPlants(self, plants: List[int], capacity: int) -> int:
        tmp_capacity = capacity
        ix = 0
        dist = 0
        while ix < len(plants):
            if tmp_capacity >= plants[ix]:
                dist += 1
                tmp_capacity -= plants[ix]
            else:
                tmp_capacity = capacity
                dist += ((ix + 1) + (ix))
                tmp_capacity -= plants[ix]
            # print(dist, ix)
            ix += 1
        return dist


s = Solution()
plants = [2,4,5,1,2]
capacity = 6
plants = [2,2,3,3]
capacity = 5
plants = [1,1,1,4,2,3]
capacity = 4
plants = [7,7,7,7,7,7,7]
capacity = 8
print(s.wateringPlants(plants, capacity))