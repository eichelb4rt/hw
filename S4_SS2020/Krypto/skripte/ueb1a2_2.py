m=15
i=0
for b in range(1,m):
    if (b**2)%m == 1:
        for c in range(0,m):
            if (c*(b+1))%m == 0:
                print(b,c)
                i+=1
print(i)