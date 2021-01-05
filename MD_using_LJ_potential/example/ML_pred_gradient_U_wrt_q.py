import torch
from torch.autograd import grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class HNN(nn.Module): #define forward that you call on the torch.nn.Module 

    def __init__(self,n_particle):
        super(HNN,self).__init__()

        in1 = 4*n_particle
        out1 = 256*n_particle
        out2 = 256*n_particle
        out3 = 1

        self.Linear_kinetic = nn.Sequential(
                nn.Linear(in1,out1),
                nn.Tanh(),
                nn.Linear(out1,out2),
                nn.Tanh(),
                nn.Linear(out2,out3)
                )

        self.Linear_potential = nn.Sequential(
                nn.Linear(in1,out1),
                nn.Tanh(),
                nn.Linear(out1,out2),
                nn.Tanh(),
                nn.Linear(out2,out3)
                )

    def forward(self,q_list,p_list):

        print('=== first input === ')
        print(q_list,p_list)

        pq = torch.cat((q_list,p_list),dim=1)
        U = self.Linear_potential(pq)
        dpdt = get_dpdt(U,q_list,p_list)
        p_list = p_list + dpdt*tau/2

        print('=== q, update p === ')
        print(q_list,p_list)

        pq = torch.cat((q_list, p_list), dim=1)
        K = self.Linear_kinetic(pq)
        dqdt = get_dqdt(K,q_list,p_list)
        q_list = q_list + dqdt*tau

        print('=== update q, p === ')
        print(q_list,p_list)

        pq = torch.cat((q_list, p_list), dim=1)
        U = self.Linear_potential(pq)
        dpdt = get_dpdt(U,q_list,p_list)
        p_list = p_list + dpdt*tau/2

        print('=== q, update p === ')
        print(q_list,p_list)

        if dpdt.shape != q.shape or dpdt.shape != p.shape:
            print('error shape not match ')

        return q_list, p_list

def get_dqdt(K,q_list,p_list):
    dqdt =    grad(K,p_list,create_graph=True,grad_outputs=torch.ones_like(K),allow_unused=False)[0]
    return dqdt

def get_dpdt(U,q_list,p_list):
    dpdt = -1*grad(U,q_list,create_graph=True,grad_outputs=torch.ones_like(U),allow_unused=False)[0]
    return dpdt

batch = 3
n_particle = 2
q0 = torch.tensor([[1.,2.,1.,1.],[1.,1.,2.,3.],[2.,3.,1.,1.]], requires_grad=True) # batch 3 , q = q1,q2,=> q1x, q1y, q2x,q2y
p0 = torch.tensor([[.2,.3,.1,.1],[.1,.1,.2,.2],[.4,.3,.5,.6]], requires_grad=True) # batch 3 , p = px, py 

q_gt = torch.tensor([[1.5,2.2,1.2,1.3],[.95,.8,2.2,3.6],[1.4,2.7,1.,1.]], requires_grad=True) # batch 3 , q = q1,q2,=> q1x, q1y, q2x,q2y
p_gt = torch.tensor([[.6,.1,-.1,.2],[-.1,.4,-.3,.7],[-.1,.0,-.1,1.]], requires_grad=True) # batch 3 , p = px, py 

tau = 0.01

hamiltonian = HNN(n_particle)
print(hamiltonian)
nepoch = 10
lr = 0.001

#optimizer on multi-NN
opt = optim.Adam(hamiltonian.parameters(),lr) 

hamiltonian.train()

for e in range(nepoch):
    q = q0
    p = p0
    q,p = hamiltonian(q,p)

    # setting gradient to zeros
    opt.zero_grad()

    loss = (torch.sum((p-p_gt)*(p-p_gt))+torch.sum((q-q_gt)*(q-q_gt)))/(batch*n_particle)

    print('=== loss data ===')
    print(loss.data)
    # backward propagation
    loss.backward()

    # update the gradient to new gradients
    opt.step()

    if e%(nepoch//10)==0: print('e ',e,'loss ',loss)





