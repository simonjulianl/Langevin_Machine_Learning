from checkpoint import checkpoint

import torch
import torch.nn as nn
import torch.optim as optim

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc1 = nn.Linear(1,1)

class stupid:
    def __init__(self):
        self.n1 = net()
        self.n2 = net()
        self.netlist = [self.n1,self.n2]

    def net_parameters(self):
        param = list(self.n1.parameters())+list(self.n2.parameters())
        return param

    def change_weight(self):
        self.n1.fc1.weight.data = self.n1.fc1.weight.data + torch.tensor([[1.0]])
        self.n2.fc1.weight.data = self.n2.fc1.weight.data + torch.tensor([[283.]])


if __name__=='__main__':

    hk = stupid()
    opt = optim.Adam(hk.net_parameters(),0.1)

    print(hk.n1.state_dict())
    print(hk.n2.state_dict())

    ckpt = checkpoint(hk.netlist,opt)
 
    ckpt.save_checkpoint('outfile.pth')

    print('after saving and then change')
    hk.change_weight()
    print(hk.n1.state_dict())
    print(hk.n2.state_dict())


    ckpt.load_checkpoint('outfile.pth')

    print('after reload ')
    print(hk.n1.state_dict())
    print(hk.n2.state_dict())
