import torch

L = torch.randn(5)
idx = torch.tensor([0,2])
# idx =[0,2]
idx = idx.tolist()
Lp = L[idx]
print('L ',L)
print('Lp ',Lp)

oriset = set(range(len(L)))
idxset = set(idx)
cidxset = oriset-idxset
print('cidxset',cidxset)
Lq = L[list(cidxset)]

print('ori set ',oriset)
print('idx set ',idxset)
print('cidx set ',cidxset)
print('Lq ',Lq)