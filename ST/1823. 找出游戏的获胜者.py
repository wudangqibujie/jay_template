class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        nums = [i for i in range(1, n + 1)]
        ids = 0
        for _ in range(n - 1):
            flg = 1
            while flg < k:
                ids += 1
                if nums[ids % n] is None:
                    continue
                flg += 1
            nums[ids % n] = None
            while nums[ids % n] is None:
                ids += 1
        for i in nums:
            if i is not None:
                return i


s = Solution()
print(s.findTheWinner(1, 1))