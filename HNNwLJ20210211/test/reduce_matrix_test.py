import torch

if __name__ == '__main__':

    q_list = torch.tensor([[[0., 0.],[2., 4.],[6., 8.]],[[1., 2.],[0., 0.],[6., 8.]],[[0., 0.],[2., 4.],[6., 8.]]])
    print(q_list)
    del_q_list = torch.delete()