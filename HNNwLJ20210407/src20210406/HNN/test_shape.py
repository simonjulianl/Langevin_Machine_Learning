import torch

nsamples = 1
nparticle = 2
DIM = 2

torch.manual_seed(3137)

r1 = torch.rand([nsamples,nparticle,nparticle,DIM])
print('r1',r1)
r2 = torch.clone(r1)
print('r2',r2)
dr = torch.diagonal(r1,0,1,2)  # take out the [:,i,i,:] element
print('dr', dr)
print('r1',r1)
rr = dr.fill_(0)
# make them zero
print('r1',r1)
print('rr', rr)

for i in range(nparticle):     # slow way of making [:,i,i,:] element zero
    r2[:,i,i,:] = 0

print('r1 ',r1)
print('r2 ',r2)

err = r1-r2  # compare answers

print('err ',err)



'''
dq_list = torch.ones([nsamples,nparticle,nparticle,DIM])
dp_list = torch.ones([nsamples,nparticle,nparticle,DIM])
ta_list = torch.ones([nsamples,nparticle,nparticle,1])

dq_list = torch.fill_(dq_list,1)
dp_list = torch.fill_(dp_list,2)
ta_list = torch.fill_(ta_list,3)

x = torch.cat((dq_list,dp_list,ta_list),dim=3)

print('x shape ',x.shape)

x = torch.reshape(x,(nsamples*nparticle*nparticle,5))

print('x shape 2 ',x.shape)

print('x ',x)
'''

