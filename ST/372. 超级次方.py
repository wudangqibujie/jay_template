
a = 2147483647
a = a % 1337
lst_mod = a
b = [2, 0, 0]
b = [str(i) for i in b]
for _ in range(int(''.join(b)) - 1):
    lst_mod = (lst_mod * a) % 1337
print(lst_mod)