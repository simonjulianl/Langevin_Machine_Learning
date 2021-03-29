import torch
import matplotlib.pyplot as plt

def build_gridpoint(npixels, boxsize, DIM):

    xvalues = torch.arange(0, npixels, dtype=torch.float64)
    xvalues = xvalues - 0.5 * npixels
    yvalues = torch.arange(0, npixels, dtype=torch.float64)
    yvalues = yvalues - 0.5 * npixels
    gridx, gridy = torch.meshgrid(xvalues * (boxsize / npixels), yvalues * (boxsize / npixels))
    grid_list = torch.stack([gridx, gridy], dim=-1)
    grid_list = grid_list.reshape((-1, DIM))

    return grid_list

def show_grid_nparticles(grid_list, q_list):

    #for i in range(MD_parameters.nsamples):
    for i in range(1):

        plt.plot(grid_list[:,0], grid_list[:,1], marker='.', color='k', linestyle='none', markersize=12)
        plt.plot(q_list[:, 0], q_list[:, 1], marker='x', color='r', linestyle='none', markersize=12)
        plt.show()
        plt.close()

def euclidean_distance(vector1, vector2):
    # print('distance', vector1, vector2)
    # print('distance', vector1.shape, vector2.shape)
    return torch.sqrt(torch.sum(torch.pow(vector1 - vector2, 2), dim=1))

if __name__ == '__main__':

    seed = 9372211
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
    # torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    nsamples = 1
    nparticle = 2
    npixels = 10
    boxsize = 2.
    DIM = 2
    grid_interval = boxsize / npixels
    print('grid interval', grid_interval)

    grid = build_gridpoint(npixels, boxsize, DIM)
    # print('grid', grid)
    # U1 = torch.rand(2, npixels, npixels)
    # # print(U1)
    # U2 = torch.rand(2, npixels, npixels)
    # # print(U2)
    # U = torch.stack([U1, U2], dim=0)
    # print('U', U)
    # print(U.shape)

    U = torch.rand(1, 2, npixels, npixels)
    print('U', U)
    print(U.shape)

    # q_list = [[-0.5, -0.5],[0.5, 0.5]]
    #q_list = [[[-0.95, -0.95],[0.5, 0.5]], [[-0.5, -0.5],[0.5, 0.5]]]
    q_list = [[[-0.95, -0.95],[0.5, 0.5]]]

    _q_list = torch.tensor(q_list, dtype=torch.float64)
    print('q list', _q_list.shape)
    # show_grid_nparticles(grid, _q_list)

    force4cnn = torch.zeros((nsamples, nparticle, DIM))

    for i in range(nparticle):

        print(i)
        print('q_list', _q_list[:,i])

        k_nearest_up_left = torch.floor(_q_list[:,i] / grid_interval ) * grid_interval

        k_nearest_up_left_x = k_nearest_up_left[:,0]
        k_nearest_up_left_x = k_nearest_up_left_x.reshape(-1,1)
        k_nearest_up_left_y = k_nearest_up_left[:,1]
        k_nearest_up_left_y = k_nearest_up_left_y.reshape(-1,1)

        k_nearest_up_right = torch.cat([k_nearest_up_left_x, k_nearest_up_left_y + grid_interval], dim=1)
        k_nearest_down_left = torch.cat([k_nearest_up_left_x + grid_interval, k_nearest_up_left_y], dim=1)
        k_nearest_down_right = torch.cat([k_nearest_up_left_x + grid_interval, k_nearest_up_left_y + grid_interval], dim=1)

        print('k_nearest_up_left, k_nearest_up_right, k_nearest_down_left, k_nearest_down_right')
        print(k_nearest_up_left, k_nearest_up_right, k_nearest_down_left, k_nearest_down_right)
        print(k_nearest_up_left.shape, k_nearest_up_right.shape, k_nearest_down_left.shape, k_nearest_down_right.shape)

        k_nearest = torch.stack([k_nearest_up_left, k_nearest_up_right, k_nearest_down_left, k_nearest_down_right]) # 4_nearest x samples x DIM
        print('k nearest')
        print(k_nearest)
        print(k_nearest.shape)

        k_nearest_up_left_distance    = euclidean_distance(_q_list[:, i], k_nearest[0])
        k_nearest_up_right_distance   = euclidean_distance(_q_list[:, i], k_nearest[1])
        k_nearest_down_left_distance  = euclidean_distance(_q_list[:, i], k_nearest[2])
        k_nearest_down_right_distance = euclidean_distance(_q_list[:, i], k_nearest[3])

        print(k_nearest_up_left_distance, k_nearest_up_right_distance, k_nearest_down_left_distance, k_nearest_down_right_distance)

        k_nearest_coord = ( boxsize / 2. + k_nearest ) / grid_interval
        kind = torch.round(k_nearest_coord).long() # 4_nearest x nsamples x DIM

        print('k nearest coord')
        print(k_nearest_coord)
        print(kind)
        print(kind.shape)
        print(kind[0, :, 0], kind[0, :, 1])
        print(kind[1, :, 0], kind[1, :, 1])
        print(kind[2, :, 0], kind[2, :, 1])
        print(kind[3, :, 0], kind[3, :, 1])

        z_distance = 1. / k_nearest_up_left_distance + 1. / k_nearest_up_right_distance + 1. / k_nearest_down_left_distance + 1. / k_nearest_down_right_distance
        print('z distance', z_distance)
        print('U', U.shape)

        predict_app = []
        for z in range(nsamples):

            print('z', z)
            predict_up_left = U[z, :, kind[0, :, 0][z], kind[0, :, 1][z]]
            predict_up_right = U[z, :, kind[1, :, 0][z], kind[1, :, 1][z]]
            predict_down_left = U[z, :, kind[2, :, 0][z], kind[2, :, 1][z]]
            predict_down_right = U[z, :, kind[3, :, 0][z], kind[3, :, 1][z]]

            print(predict_up_left.shape, predict_up_right.shape, predict_down_left.shape, predict_down_right.shape)
            print(predict_up_left, predict_up_right, predict_down_left, predict_down_right)
            predict_cat = torch.stack((predict_up_left, predict_up_right, predict_down_left, predict_down_right))
            predict_app.append(predict_cat)

        predict_k_nearest_force = torch.stack(predict_app)  # sample x 4_nearest x DIM
        print(predict_k_nearest_force)
        print(predict_k_nearest_force.shape)

        predict_each_particle_app = []

        for z in range(nsamples):

            predict_each_particle = (1. / z_distance[z]) * (1. / k_nearest_up_left_distance[z] * predict_k_nearest_force[z,0]
                                                         + 1. / k_nearest_up_right_distance[z] * predict_k_nearest_force[z,1]
                                                         + 1. / k_nearest_down_left_distance[z] * predict_k_nearest_force[z,2]
                                                         + 1. / k_nearest_down_right_distance[z] * predict_k_nearest_force[z,3])

            predict_each_particle_app.append(predict_each_particle)

        predict_each_particle = torch.stack(predict_each_particle_app)
        print(predict_each_particle)
        print(predict_each_particle.shape)

        force4cnn[:, i] = predict_each_particle
        print('force', force4cnn[:, i])

    print(force4cnn)