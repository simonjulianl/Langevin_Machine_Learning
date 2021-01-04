import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NN(nn.Module): #define forward that you call on the torch.nn.Module 

    def __init__(self,n_particle):
        super(NN,self).__init__()

        in1 = 1*n_particle

        #weight
        self.w = torch.randn(in1,n_particle)
        print(self.w)

    def forward(self,x):

        out = torch.matmul(x, self.w)

        return out


n_particle = 1
q = torch.tensor([1.], requires_grad=True) # batch 3 , q = q1,q2,=> q1x, q1y, q2x,q2y
p = torch.tensor([1.], requires_grad=True) # batch 3 , p = px, py

q_gt = torch.tensor([1.], requires_grad=True) # batch 3 , q = q1,q2,=> q1x, q1y, q2x,q2y
p_gt = torch.tensor([1.], requires_grad=True) # batch 3 , p = px, py 

tau = 0.01

net = NN(n_particle)
print(net)
nepoch = 100 
lr = 0.1

#optimizer on multi-NN
opt = optim.Adam(net.parameters(),lr)
net.train()

for e in range(nepoch):

    force = net(q)
    p = p + force*tau/2

    q = q + p*tau

    force = net(q)
    p = p + force*tau/2

    # setting gradient to zeros
    opt.zero_grad()
    loss = (torch.sum((p-p_gt)*(p-p_gt))+torch.sum((q-q_gt)*(q-q_gt)))/(n_particle)
    # backward propagation
    loss.backward()
    # update the gradient to new gradients
    opt.step()
    if e%(nepoch//10)==0: print('e ',e,'loss ',loss)



# if dpdt.shape != q.shape or dpdt.shape != p.shape:
#     print('error shape not match ')

