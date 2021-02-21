import torch

if __name__ == '__main__':

    q_list = [[[3,2],[2.2,1.21]],[[1,3],[2,1]]]
    p_list = [[[0.1,0.4],[0.1,0.1]],[[0.2,0.4],[0.3,0.1]]]

    q_list, p_list = torch.tensor([q_list,p_list])

    epochs = 10

    for i in range(epochs):
        # shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(i)

        idx = torch.randperm(q_list.shape[0], generator=g)
        q_list, p_list = q_list[idx], p_list[idx]
        print('idx',idx)
        # print(q_list, p_list)
        # print(q_list[idx[0]], q_list[idx[0]].shape)
        #
