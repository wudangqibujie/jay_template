from typing import List

class Solution:
    def subdomainVisits(self, cpdomains: List[str]) -> List[str]:
        log = dict()
        for ch in cpdomains:
            times, name = ch.split(' ')
            times = int(times)
            names = name.split('.')
            for i in range(1, len(names) + 1):
                sub_name = '.'.join(names[-i:])
                if sub_name in log:
                    log[sub_name] += times
                else:
                    log[sub_name] = times
        return [str(v) + ' ' + k for k, v in log.items()]

s = Solution()
cpdomains = ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
print(s.subdomainVisits(cpdomains))