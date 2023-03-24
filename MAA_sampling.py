import random

def random_sampling(n):
    
    r = [None] * (n+1)
    rr = [None] * n
    a_list = [0,1]
    
    for i in range(n): 
        rr[i] = random.uniform(0, 1)
        a_list.insert(i+1, rr[i])
        a_list.sort()
    
    for j in range(len(a_list)-1):
        r[j] = a_list[j+1] - a_list[j]
    
    return r

r = random_sampling(2)
print('The random vector is: ' + str(r) + ' and sums to: ' + str(sum(r)))