import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class NN(nn.Module): #define forward that you call on the torch.nn.Module 

    def __init__(self):
        super(NN,self).__init__()

        in1 = 1
        self.fc = nn.Linear(in1,1,bias=False)

    def forward(self,x):

        out = self.fc(x)

        return out

# ex) q = t , qdot = p = 1
tau = 1

seed =  937162211

torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
# torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

q0 = torch.tensor([1.], requires_grad=True)
qdot0 = torch.tensor([1.], requires_grad=True)

q_gt = torch.tensor([2.] )
qdot_gt = torch.tensor([1.] )
# print(q_gt)
# print(p_gt)
# quit()
net = NN()
print(net)

nepoch = 10 # 100000
lr = 0.0001

#optimizer on multi-NN
opt = optim.Adam(net.parameters(),lr)
net.train()

for e in range(nepoch):

    q = q0
    qdot = qdot0
    print('============== ')
    print('epoch {}'.format(e))
    print('============== ')

    print('=== first input === ')
    print(q)
    f_pred = net(q) #f_w

    print('=== First calc force === ')
    print('======== weight ======== ')
    print(net.fc.weight)
    print('========= bias ========= ')
    print(net.fc.bias)
    print('========= force ========= ')
    print(f_pred)

    qdot = qdot + f_pred * tau/2
    print('=== update p ===')
    print(qdot)
    print('================')

    q = q + qdot * tau
    print('=== update q ===')
    print(q)
    print('================')

    print('=== second input === ')
    print(q)

    f_pred = net(q)
    print('== second calc force === ')
    print('======== weight ======== ')
    print(net.fc.weight)
    print('========= bias ========= ')
    print(net.fc.bias)
    print('========= force ========= ')
    print(f_pred)
    print('============== ')

    qdot = qdot + f_pred * tau/2
    print('=== update p ===')
    print(qdot)
    print('================')

    # setting gradient to zeros
    opt.zero_grad()
    loss = (torch.sum((qdot-qdot_gt)*(qdot-qdot_gt))+torch.sum((q-q_gt)*(q-q_gt)))
    # backward propagation
    loss.backward()

    # update the gradient to new gradients
    opt.step()
    if e%(nepoch//10)==0: print('e ',e,'loss ',loss.item())


# if dpdt.shape != q.shape or dpdt.shape != p.shape:
#     print('error shape not match ')

