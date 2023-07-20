class Solution:
    def numTilePossibilities(self, tiles: str) -> int:
        rslt = []
        def search(now, candi):
            if not candi:
                if now not in rslt:
                    rslt.append(now)
                return
            if now not in rslt and now != '':
                rslt.append(now)
            for ix, i in enumerate(candi):
                new_candi = [i for jx, i in enumerate(candi) if jx != ix]
                search(now + i, new_candi)

        search('', tiles)
        # print(rslt)
        return len(rslt)


s = Solution()
tiles = "V"
print(s.numTilePossibilities(tiles))