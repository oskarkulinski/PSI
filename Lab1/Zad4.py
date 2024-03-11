n = int(input("Length?:"))

for i in range(n):
    print ("|....", end="")
print("|")
print("0", end="")
for i in range(1, n+1):
    print("%5s" % i, end="")
print()