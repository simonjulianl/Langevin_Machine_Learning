import torch

class interpolator: # HK

    _obj_count = 0

    def __init__(self):

        interpolator._obj_count += 1
        assert (interpolator._obj_count == 1),type(self).__name__ + " has more than one object"


    def inverse_interpolator(self, predict, phase_space):

        nsamples, channels, gridL, _ = predict.shape
        # predict shape is [nsamples, channels=(x,y), gridx, dridy]

        q_list = phase_space.get_q()
        _, nparticle, DIM = q_list.shape
        # q_list shape is [nsamples, nparticle, DIM]

        boxsize = phase_space.get_boxsize()

        grid_interval = boxsize / gridL

        nn_nparticle_app = []
        nn_coord_nparticle_app = []

        for i in range(nparticle):
            '''
            Distance btw particle and 4-nearest neighbor grids
            nn_up_left_distance      : distance btw up left grid and particle
            nn_up_right_distance     : distance btw up right grid and particle
            nn_down_left_distance    : distance btw down left grid and particle
            nn_down_right_distance   : distance btw down right grid and particle
            '''

            nn = self.nn_grids(q_list[:, i, :], grid_interval)
            # q_list shape is [nsamples, DIM=(x,y)]
            # nn.shape is [nsamples, 4-nn grids, 2=(x,y)]

            nn_up_left_distance = self.euclidean_distance(q_list[:, i, :], nn[:, 0, :])
            nn_up_right_distance = self.euclidean_distance(q_list[:, i, :], nn[:, 1, :])
            nn_down_left_distance = self.euclidean_distance(q_list[:, i, :], nn[:, 2, :])
            nn_down_right_distance = self.euclidean_distance(q_list[:, i, :], nn[:, 3, :])
            #shape is [nsamples, DIM]

            nn_distance_cat = torch.stack((nn_up_left_distance, nn_up_right_distance,
                                                  nn_down_left_distance, nn_down_right_distance), dim=-1)
            # nn_distance_cat.shape is [nsamples, 4-nn girds]

            nn_nparticle_app.append(nn_distance_cat)

            nn_ind = self.nn_coord(nn, grid_interval, boxsize)
            # nn_ind.shape is [nsamples, 4-nn grids, DIM]

            nn_coord_nparticle_app.append(nn_ind)

        nn_distance_nparticle = torch.stack(nn_nparticle_app, dim=1)
        # shape is [nsamples, nparticle, 4-nn grids]
        nn_coord_nparticle = torch.stack(nn_coord_nparticle_app, dim=1)
        # shape is [nsamples, nparticle, 4-nn grids, DIM]
        print('nn_distance_nparticle', nn_distance_nparticle)
        print('nn_coord_nparticle', nn_coord_nparticle)

        predict_app = []

        for z in range(nsamples): # take 4-nn forces
            '''
            find predicted phi fields of 4-nearest neighbor grids
            predict_up_left      : predicted phi field of up left grid  
            predict_up_right     : predicted phi field of up right grid  
            predict_down_left    : predicted phi field of down left grid 
            predict_down_right   : predicted phi field of down right grid  
            '''

            # predict.shape is [nsamples, channels=(x,y), gridx, dridy]
            predict_up_left = predict[z, :, nn_coord_nparticle[z, :, 0, 0], nn_coord_nparticle[z, :, 0, 1]]
            predict_up_right = predict[z, :, nn_coord_nparticle[z, :, 1, 0], nn_coord_nparticle[z, :, 1, 1]]
            predict_down_left = predict[z, :, nn_coord_nparticle[z, :, 2, 0], nn_coord_nparticle[z, :, 2, 1]]
            predict_down_right = predict[z, :, nn_coord_nparticle[z, :, 3, 0], nn_coord_nparticle[z, :, 3, 1]]
            #shape is [DIM, nparticle]

            predict_cat = torch.stack((predict_up_left, predict_up_right, predict_down_left, predict_down_right), dim=-1)
            # shape is [DIM, nparticle, 4-nn grids]
            print('predict', predict_cat.shape)
            predict_app.append(predict_cat)

        predict_nn_force = torch.stack(predict_app)
        # predict_nn_force.shape is [nsmples, DIM, nparticle, 4-nn grids]

        print('predict_nn_force', predict_nn_force)

        ###################################################################
        ################ start interpolate calculation ####################
        ###################################################################

        z_l = torch.sum(1. / nn_distance_nparticle, dim=-1)
        # shape is [nsamples, nparticle]
        z_l = z_l.unsqueeze(dim=1)
        # shape is [nsamples, 1, nparticle]

        nn_distance_nparticle_unsqueeze = nn_distance_nparticle.unsqueeze(dim=1)
        # shape is [nsamples, 1 ,nparticle ,4-nn grids]

        predict_each_particle = 1. / z_l * (torch.sum(1. / nn_distance_nparticle_unsqueeze * predict_nn_force, dim=-1))
        # shape is [nsmples, DIM, nparticle ]
        predict_each_particle = predict_each_particle.permute((0, 2, 1))
        # shape is [nsmples, nparticle , DIM]

        if (nn_distance_nparticle_unsqueeze < 0.001).any():
            index = torch.where(nn_distance_nparticle < 0.001)
            # nn_distance_nparticle.shape is [nsamples, nparticle, 4-nn grids]

            predict_k_vv_nearest_force = predict_nn_force[index[0], :, index[1], index[2]]
            # predict_nn_force.shape is [nsmples, DIM, nparticle, 4-nn grids]

            print(predict_k_vv_nearest_force.shape)
            predict_each_particle[index[0], index[1]] = predict_k_vv_nearest_force
        ###################################################################
        ################  end  interpolate calculation ####################
        ###################################################################

        return predict_each_particle

    def euclidean_distance(self, vector1, vector2):
        ''' function to measure distance between nearest grid and particle
        :parameter
        vector1 : shape is [nsamples, DIM]
        vector2 : shape is [nsamples, DIM]
        '''

        return torch.sqrt(torch.sum(torch.pow(vector1 - vector2, 2), dim=1))  # sum DIM that is dim=1

    def nn_grids(self, q_list, grid_interval):

        ''' function to find four nearest neighbor grids of particle

        :parameter
        q_list : shape is [nsamples, DIM]
        nn_up_left      : up left in particle
        nn_up_right     : up right in particle
        nn_down_left    : down left in particle
        nn_down_right   : down right in particle
        '''

        nn_up_left = torch.floor(q_list / grid_interval) * grid_interval

        nn_up_left_x = nn_up_left[:, 0]
        nn_up_left_x = nn_up_left_x.reshape(-1, 1)

        nn_up_left_y = nn_up_left[:, 1]
        nn_up_left_y = nn_up_left_y.reshape(-1, 1)

        nn_up_right = torch.cat([nn_up_left_x, nn_up_left_y + grid_interval], dim=1)
        nn_down_left = torch.cat([nn_up_left_x + grid_interval, nn_up_left_y], dim=1)
        nn_down_right = torch.cat([nn_up_left_x + grid_interval, nn_up_left_y + grid_interval], dim=1)

        nn = torch.stack([nn_up_left, nn_up_right, nn_down_left, nn_down_right], dim=1)
        # shape is [nsamples, 4-nn grids, DIM=(x,y)]

        return nn

    def nn_coord(self, nn, grid_interval, boxsize):

        ''' function to find index of phi_fields about 4 nn grids '''

        nn_coord = ( boxsize / 2. + nn) / grid_interval
        nn_ind = torch.round(nn_coord).long()
        # nn_ind.shape is [nsamples, 4-nn grids, DIM]

        return nn_ind



