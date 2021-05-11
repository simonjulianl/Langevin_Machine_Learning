import numpy as np
import scipy.interpolate

import torch

dtype = torch.cuda.FloatTensor
dtype_long = torch.cuda.LongTensor


def bilinear_interpolate_torch(im, x, y):
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1

    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    print(x0,x1,y0,y1)

    Ia = im[y0, x0][0]
    Ib = im[y1, x0][0]
    Ic = im[y0, x1][0]
    Id = im[y1, x1][0]

    print(Ia,Ib,Ic,Id)
    wa = (x1.type(dtype) - x) * (y1.type(dtype) - y)
    wb = (x1.type(dtype) - x) * (y - y0.type(dtype))
    wc = (x - x0.type(dtype)) * (y1.type(dtype) - y)
    wd = (x - x0.type(dtype)) * (y - y0.type(dtype))
    print(wa,wb,wc,wd)

    return torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
        torch.t(Id) * wd)

def bilinear_interpolate_scipy(image, x, y):
    x_indices = np.arange(image.shape[0])
    y_indices = np.arange(image.shape[1])
    interp_func = scipy.interpolate.interp2d(x_indices, y_indices, image, kind='linear')
    return interp_func(x,y)

# Make small sample data that's easy to interpret
image = np.ones((5,5))
image[3,3] = 4
image[3,4] = 3

sample_x, sample_y = np.asarray([3.2]), np.asarray([3.4])
print(image)
print(sample_x, sample_y)

print ("scipy result:", bilinear_interpolate_scipy(image, sample_x, sample_y))

image = torch.unsqueeze(torch.FloatTensor(image).type(dtype),2)
sample_x = torch.FloatTensor([sample_x]).type(dtype)
sample_y = torch.FloatTensor([sample_y]).type(dtype)

print ("torch result:", bilinear_interpolate_torch(image, sample_x, sample_y))