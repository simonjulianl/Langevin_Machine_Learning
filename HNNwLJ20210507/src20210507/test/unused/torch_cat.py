import torch

if __name__ == '__main__':

    torch_tensor1 =  torch.rand((1))
    torch_tensor2 = torch.rand((1))

    torch_cat = torch.cat((torch_tensor1, torch_tensor2),dim=0)
    print(torch_cat)
    torch_cat2 = torch.cat((torch_cat,torch_tensor1),dim=0)
    print(torch_cat2)