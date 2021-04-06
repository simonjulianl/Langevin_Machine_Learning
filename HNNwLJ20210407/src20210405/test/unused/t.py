import torch

if __name__=='__main__':

    a = torch.randn([10])
    print('a',a)

    u = (a>1)

    print(u)
