import torch

size = 2
channels = 2
batch_size = 1
a = torch.tensor([[[[1.,2.],[3.,4.]],[[11.,22.],[33.,44.]]]])
print(a.shape)

v =torch.zeros(size,)
v[0] = 1

v_flip = torch.flip(v, dims=(0,))

v = torch.unsqueeze(v, dim=0)
v_flip = torch.unsqueeze(v_flip, dim=0)

b = torch.cat([v_flip, torch.eye(size), v])
b_t = torch.transpose(b, 0, 1)

print('b', b.shape)
print('b_t', b_t.shape)

b = b.unsqueeze(0).repeat(batch_size, 1, 1)
b_t = b_t.unsqueeze(0).repeat(batch_size, 1, 1)
b_conv = b.unsqueeze(1).repeat(1, channels, 1, 1)
b_conv_t = b_t.unsqueeze(1).repeat(1, channels, 1, 1)

print(a.shape)
print(b_conv.shape)
print(a.dtype, b_conv.dtype)
print(torch.matmul(b_conv,a))

x = torch.matmul(torch.matmul(b_conv,a),b_conv_t)

print(x)

