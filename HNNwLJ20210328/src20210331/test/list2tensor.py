import torch

if __name__ == '__main__':

    q_app = []
    for i in range(2):
        q = torch.tensor([[[2,3],[5,2]],[[9,6],[7,4]]])
        q_app.append(q)

    print(q_app)

    q_tensor = torch.cat(q_app)
    print(q_tensor)
    print(q_tensor.shape)