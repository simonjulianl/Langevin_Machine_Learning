import torch

if __name__ == '__main__':

    a = torch.tensor([[[3,7],[9,1]], [[2,3],[5,3]],[[5,2],[1,5]]])
    print(a)
    print(a.shape)

    mask = a  >  3
    a[mask] = 3

    print(a)

    for z in range(a.shape[0]):

        mask = a[z] > 3
        a[z][mask] = 3

    print(a)
