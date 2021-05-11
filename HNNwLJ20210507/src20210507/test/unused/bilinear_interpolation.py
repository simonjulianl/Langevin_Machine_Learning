import torch
dtype = torch.FloatTensor
dtype_long = torch.LongTensor


def bilinear_interpolate_torch(im, x, y):
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1

    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0][0]
    Ib = im[y1, x0][0]
    Ic = im[y0, x1][0]
    Id = im[y1, x1][0]

    wa = (x1.type(dtype) - x) * (y1.type(dtype) - y)
    wb = (x1.type(dtype) - x) * (y - y0.type(dtype))
    wc = (x - x0.type(dtype)) * (y1.type(dtype) - y)
    wd = (x - x0.type(dtype)) * (y - y0.type(dtype))

    return torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
        torch.t(Id) * wd)

if __name__ == '__main__':
    # Do high dimensional bilinear interpolation in numpy and PyTorch
    W, H, C = 5, 5, 2
    image = torch.randn(W, H, C)
    print(image)
    num_samples = 4
    samples_x, samples_y = torch.randn(num_samples) * (W - 1), torch.randn(num_samples) * (H - 1)
    print(samples_x, samples_y)
    print (bilinear_interpolate_torch(image, samples_x, samples_y))

    image = torch.from_numpy(image).type(dtype)
    samples_x = torch.FloatTensor([samples_x]).type(dtype)
    samples_y = torch.FloatTensor([samples_y]).type(dtype)

    print (bilinear_interpolate_torch(image, samples_x, samples_y))