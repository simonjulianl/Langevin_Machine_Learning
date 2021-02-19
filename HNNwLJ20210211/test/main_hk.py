import numpy as np
import torch

# ============================================
#def to_reshape(k):
#
#    k_shape = list(k.shape)
#
#    k1 = k_shape[0]*k_shape[1]
#    k2 = k_shape[2]
#    ks = torch.reshape(k,(k1,k2))
#    return ks

# ============================================
# this is a slow for loop process but do this only once
# for each shape of m
def get_indices(s):

    n = s[0]
    m = torch.ones(s)
    for i in range(n):
#        for j in range(n):
#            if i==j:
                m[i,i,:] = 0

#    mr = to_reshape(m)
#    mr_len = mr.shape[0]
#    ind = []
#    for i in range(mr_len):
#        if mr[i,0] == 1:
#            ind.append(i)
#    return ind  
    ind = m.nonzero(as_tuple=True)
    return ind

# ============================================
if __name__=='__main__':


    # first get the indices using a tensor of ones, using a slow
    # for loop method
    n = 3
    ind = get_indices((n,n,2)) 

    # now use the 'fast' method to extract the required indices
    r = torch.rand(n,n,2)
#    rr = to_reshape(r)

    print('r  ',r)
#    print('rr ',rr)

    # this line of code is fast as it is only indices extraction
    flatten_r = r[ind]
    flatten_r = flatten_r.reshape((n,n-1,2)) # <--- SJ add this

    print('flatten ',flatten_r)

