import torch
dtype = torch.FloatTensor
dtype_long = torch.LongTensor

class interpolator: # HK

    _obj_count = 0

    def __init__(self):

        interpolator._obj_count += 1
        assert (interpolator._obj_count == 1),type(self).__name__ + " has more than one object"

    def bilinear_interpolate_torch(self, im, x, y, grid_interval):

        nn_x0 = torch.floor(x / grid_interval) * grid_interval
        print(nn_x0)
        nn_x1 = nn_x0 + grid_interval
        print(nn_x1)
        nn_y0 = torch.floor(y / grid_interval) * grid_interval
        print(nn_y0)
        nn_y1 = nn_y0 + grid_interval
        print(nn_y1)
        quit()

        Ia = im[nn_y0, nn_x0][0]
        Ib = im[nn_y1, nn_x0][0]
        Ic = im[y0, x1][0]
        Id = im[y1, x1][0]

        wa = (x1.type(dtype) - x) * (y1.type(dtype) - y)
        wb = (x1.type(dtype) - x) * (y - y0.type(dtype))
        wc = (x - x0.type(dtype)) * (y1.type(dtype) - y)
        wd = (x - x0.type(dtype)) * (y - y0.type(dtype))

        return torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(
            torch.t(Id) * wd)


    def inverse_interpolator(self, predict, phase_space, grid_interval):

        nsamples, channels, gridL, _ = predict.shape
        # predict shape is [nsamples, channels=(x,y), gridx, dridy]
        predict = predict[0]
        predict = predict.permute((1,2,0))
        print('predict', predict.shape)
        q_list = phase_space.get_q()
        print('q', q_list)
        _, nparticle, DIM = q_list.shape
        # q_list shape is [nsamples, nparticle, DIM]

        samples_x, samples_y = q_list[:,:,0], q_list[:,:,1]
        samples_x = torch.squeeze(samples_x,dim=0)
        samples_y = torch.squeeze(samples_y,dim=0)
        return self.bilinear_interpolate_torch(predict, samples_x, samples_y, grid_interval)


