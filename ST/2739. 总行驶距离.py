class Solution:
    def distanceTraveled(self, mainTank: int, additionalTank: int) -> int:
        val = 0
        while mainTank:
            if mainTank < 5:
                val += (mainTank * 10)
                break
            mainTank -= 5
            val += 50
            if additionalTank:
                mainTank += 1
                additionalTank -= 1
        return val


s = Solution()
mainTank = 5
additionalTank = 10
mainTank = 15
additionalTank = 2
print(s.distanceTraveled(mainTank, additionalTank))

