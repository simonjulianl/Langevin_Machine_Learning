from checkpoint import checkpoint
import torch.optim.lr_scheduler as scheduler

import torch
import torch.nn as nn
import torch.optim as optim

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc1 = nn.Linear(2,2)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        return x

class stupid:
    def __init__(self):
        self.n1 = net()
        self.n2 = net()
        self.netlist = [self.n1,self.n2]

    def net_parameters(self):
        param = list(self.n1.parameters())+list(self.n2.parameters())
        return param

    def change_weight(self):
        self.n1.fc1.weight.data = self.n1.fc1.weight.data + torch.tensor([[2.0]])
        self.n2.fc1.weight.data = self.n2.fc1.weight.data + torch.tensor([[1.0]])

if __name__ == '__main__':

    torch.manual_seed(522)

    nepochs = 10
    every_n_epoch = 2
    decay_rate = 0.9

    a = torch.rand((2,2))
    b = torch.zeros((2,2))

    print('input')
    print(a)
    print('label')
    print(b)

    hk = stupid()
    opt = optim.Adam(hk.net_parameters(), lr=0.1)
    sch = scheduler.StepLR(opt,every_n_epoch,decay_rate)

    chkpt = checkpoint(hk.netlist, opt, sch)

    # print('before save param')
    # print(hk.n1.state_dict())
    # print(hk.n2.state_dict())

    hk.n1.train()
    hk.n2.train()

    print('========= start train ===========')
    for e in range(nepochs):

        opt.zero_grad()
        pred = hk.n1(a)
        loss = (torch.sum(pred-b))
        loss.backward() # compute gradient of the loss
        opt.step() # parameter update
        sch.step()

        chkpt.save_checkpoint('outfile.pth')# hamiltonian.eval() # HK
        print('lr',opt.param_groups[0]['lr'], 'e', e+1, 'loss', loss)

    print('===================================')

    print('======== after save param ========= ')
    print(hk.n1.state_dict())
    print(hk.n2.state_dict())
    print('===================================')

    hk.change_weight()

    print('========= change weight ==========')
    print(hk.n1.state_dict())
    print(hk.n2.state_dict())
    print('=================================')

    chkpt.load_checkpoint('outfile.pth')

    print('=========== load weight ===========')
    print(hk.n1.state_dict())
    print(hk.n2.state_dict())
    print('================================')

    print('========= start retrain ===========')
    for e in range(nepochs):

        opt.zero_grad()
        pred = hk.n1(a)
        loss = (torch.sum(pred-b))
        loss.backward()
        opt.step()
        sch.step()

        print('lr', opt.param_groups[0]['lr'], 'e', e + 1, 'loss', loss)

    print('====================================')