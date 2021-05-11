import torch
dtype = torch.FloatTensor
dtype_long = torch.LongTensor

class interpolator:
    ''' interpolator class to calculate force of each particle
    from forces of 4 nearest neighbor grids  '''

    _obj_count = 0

    def __init__(self):

        interpolator._obj_count += 1
        assert (interpolator._obj_count == 1),type(self).__name__ + " has more than one object"

    def bilinear_interpolate_torch(self, im, x, y, grid_interval):
        '''
        parameter
        ------------
        im  : shape is [gridy, gridx, channels=(x,y)]
        x   : npartcles in x-axis
                shape is [1, nparticle]
        y   : nparticles in y-axis
                shape is [1, nparticle]
        grid_interval : boxsize / gridL
        '''

        x0 = torch.floor(x).type(dtype_long)
        x1 = x0 + 1

        y0 = torch.floor(y).type(dtype_long)
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, im.shape[1] - 1)
        x1 = torch.clamp(x1, 0, im.shape[1] - 1)
        y0 = torch.clamp(y0, 0, im.shape[0] - 1)
        y1 = torch.clamp(y1, 0, im.shape[0] - 1)

        # print('x', x0, x1, 'y', y0, y1)

        Ia = im[x0, y0][0]
        Ib = im[x0, y1][0]
        Ic = im[x1, y0][0]
        Id = im[x1, y1][0]
        # im[y0, x0].shape is [ 1, nparticle, DIM=(x,y) ]
        # Ia.shape is [nparticle, DIM=(x,y)]
        # print('I', Ia, Ib, Ic, Id)

        wa = (x1.type(dtype) - x) * (y1.type(dtype) - y) * (grid_interval * grid_interval)
        wb = (x1.type(dtype) - x) * (y - y0.type(dtype)) * (grid_interval * grid_interval)
        wc = (x - x0.type(dtype)) * (y1.type(dtype) - y) * (grid_interval * grid_interval)
        wd = (x - x0.type(dtype)) * (y - y0.type(dtype)) * (grid_interval * grid_interval)
        # wa.shape is [1, nparticle]
        # print('w', wa, wb, wc, wd)

        return Ia * torch.t(wa) + Ib * torch.t(wb) + Ic * torch.t(wc) + Id * torch.t(wd)


    def nsamples_interpolator(self, phi_predict, phase_space):

        nsamples, channels, gridL, _ = phi_predict.shape
        # predict shape is [nsamples, channels=(x,y), gridx, dridy]

        q_list = phase_space.get_q()
        # q_list.shape is [nsmamples, nparticle, DIM]
        # print('q_list',q_list)
        boxsize = phase_space.get_boxsize()
        grid_interval = boxsize / gridL
        # print('interval', grid_interval)
        # print('boxszie', boxsize)
        q_list_shift = q_list / grid_interval + gridL / 2
        print('q_list_shift', q_list_shift)

        phase_space.set_q(q_list_shift)

        interpolate_arr = []
        for i in range(nsamples):

            predict = phi_predict[i]

            nparticle_x, nparticle_y = q_list_shift[i,:,0], q_list_shift[i,:,1]
            # npartcle_x.shape is [nparticle]
            nparticle_x = nparticle_x.unsqueeze(dim=0)
            # nparticle_x.shape is [1, nparticle]
            nparticle_y = nparticle_y.unsqueeze(dim=0)
            # nparticle_y.shape is [1, nparticle]
            image = predict.permute((1, 2, 0))
            # image.shape is [gridx, gridy, channels=(x,y)]

            n_interpolate =  self.bilinear_interpolate_torch(image, nparticle_x, nparticle_y, grid_interval)
            # calc.shape is [nparticle, DIM=(x,y)]
            # print('n_interpolate', n_interpolate)
            interpolate_arr.append(n_interpolate)

        interpolate = torch.stack(interpolate_arr)
        # calc.shape is [nsampcles, nparticle, DIM=(x,y)]

        return interpolate

