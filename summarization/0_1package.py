import numpy as np

def knapsack(w,v,C):
    mem = np.zeros((len(w)+1,C+1))
    for i in range(1,len(w)+1):
        for j in range(1,C+1):
            if w[i-1]<=j:
                mem[i,j] = max(mem[i-1,j],mem[i-1,j-w[i-1]]+v[i-1])
            else:
                mem[i,j] = mem[i-1,j]
    return mem

v = [8,10,6,3,7,2]
w = [4,6,2,2,5,1]
C=12
print(knapsack(w,v,C))