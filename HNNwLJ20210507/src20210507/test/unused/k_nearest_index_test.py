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
        plt.plot(q_list[i, :,0], q_list[i, :, 1], marker='x', color='r', linestyle='none', markersize=12)
        plt.show()
        plt.close()

def euclidean_distance(vector1, vector2):
    # print('distance', vector1, vector2)
    # print('distance', vector1.shape, vector2.shape)
    return torch.sqrt(torch.sum(torch.pow(vector1 - vector2, 2), dim=1))  # sum DIM that is dim=1 <- nparticle x DIM

def k_nearest_grids( q_list, grid_interval):
    # to calculate distance between particle and grid
    # print('def k nearest grids')
    # print(q_list)
    k_nearest_up_left = torch.floor( q_list / grid_interval) * grid_interval

    k_nearest_up_left_x = k_nearest_up_left[:, 0]
    k_nearest_up_left_x = k_nearest_up_left_x.reshape(-1, 1)
    k_nearest_up_left_y = k_nearest_up_left[:, 1]
    k_nearest_up_left_y = k_nearest_up_left_y.reshape(-1, 1)

    k_nearest_up_right = torch.cat([k_nearest_up_left_x, k_nearest_up_left_y + grid_interval], dim=1)
    k_nearest_down_left = torch.cat([k_nearest_up_left_x + grid_interval, k_nearest_up_left_y], dim=1)
    k_nearest_down_right = torch.cat([k_nearest_up_left_x + grid_interval, k_nearest_up_left_y + grid_interval], dim=1)

    # print(k_nearest_up_left, k_nearest_up_right, k_nearest_down_left, k_nearest_down_right)
    # print(k_nearest_up_left.shape, k_nearest_up_right.shape, k_nearest_down_left.shape, k_nearest_down_right.shape)
    k_nearest = torch.stack([k_nearest_up_left, k_nearest_up_right, k_nearest_down_left, k_nearest_down_right], dim=1)  # samples x 4_nearest x DIM

    return k_nearest

def k_nearest_coord(k_nearest, grid_interval):
    # to find force k neareset grids

    k_nearest_coord = (boxsize / 2. + k_nearest) / grid_interval
    kind = torch.round(k_nearest_coord).long()  # 4_nearest x nsamples x DIM
    return kind

if __name__ == '__main__':

    seed = 9372211
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
    # torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    nsamples = 3
    nparticle = 2
    npixels = 10
    boxsize = 2.
    DIM = 2
    grid_interval = boxsize / npixels
    print('grid interval', grid_interval)

    grid = build_gridpoint(npixels, boxsize, DIM)
    # print('grid', grid)
    U1 = torch.rand(2, npixels, npixels)
    # # print(U1)
    U2 = torch.rand(2, npixels, npixels)
    # # print(U2)
    U3 = torch.rand(2, npixels, npixels)
    U = torch.stack([U1, U2, U3], dim=0)
    print('U', U)
    # print(U.shape)

    # U = torch.rand(1, 2, npixels, npixels)
    # print('U', U)
    # print(U.shape)

    # q_list = [[-0.5, -0.5],[0.5, 0.5]]
    q_list = [[[-0.95, -0.95],[0.5, 0.5]], [[-0.6001, -0.6],[0.05, 0.3]], [[-0.63, -0.2],[0.4, 0.2]]]
    # q_list = [[[-0.95, -0.95],[0.5, 0.5]]]

    _q_list = torch.tensor(q_list, dtype=torch.float64)
    print(_q_list)
    print(_q_list.shape)
    # show_grid_nparticles(grid, _q_list)

    k_nearest_nparticle_app = []
    k_nearest_coord_nparticle_app = []

    # Distance btw each particle and 4 nearest grids
    # 4 nearest girds coordinates
    for i in range(nparticle):

        # print(_q_list[:,i].shape)
        print(_q_list[:,i])  # nsamples x each particle

        k_nearest = k_nearest_grids(_q_list[:,i], grid_interval)
        print('k nearest', k_nearest)
        # print(k_nearest.shape)

        k_nearest_up_left_distance    = euclidean_distance(_q_list[:, i], k_nearest[:,0])
        k_nearest_up_right_distance   = euclidean_distance(_q_list[:, i], k_nearest[:,1])
        k_nearest_down_left_distance  = euclidean_distance(_q_list[:, i], k_nearest[:,2])
        k_nearest_down_right_distance = euclidean_distance(_q_list[:, i], k_nearest[:,3])

        print('nearest distance')
        print(k_nearest_up_left_distance, k_nearest_up_right_distance, k_nearest_down_left_distance, k_nearest_down_right_distance)

        k_nearest_distance_cat = torch.stack((k_nearest_up_left_distance, k_nearest_up_right_distance, k_nearest_down_left_distance, k_nearest_down_right_distance),dim=-1)
        k_nearest_nparticle_app.append(k_nearest_distance_cat)

        kind = k_nearest_coord(k_nearest, grid_interval)
        k_nearest_coord_nparticle_app.append(kind)

    k_nearest_distance_nparticle = torch.stack(k_nearest_nparticle_app, dim=1) #nsample x nparticle x k_nearest
    k_nearest_coord_nparticle = torch.stack(k_nearest_coord_nparticle_app, dim=1) #nsample x nparticle x k_nearest x DIM

    print('distance cat', k_nearest_distance_nparticle)
    # print(k_nearest_distance_nparticle.shape)
    print('cood', k_nearest_coord_nparticle)
    print(k_nearest_coord_nparticle.shape)
    quit()
    # print(U.shape)
    # print(k_nearest_coord_nparticle[:,:,:,0].shape,  k_nearest_coord_nparticle[:,:,:,1].shape)

    predict_app = []
    # take 4 nearest forces
    for z in range(nsamples):

        print('z', z)
        # shape = DIM x nparticles
        predict_up_left = U[z, :, k_nearest_coord_nparticle[z,:,0,0], k_nearest_coord_nparticle[z,:,0,1]]
        predict_up_right = U[z, :, k_nearest_coord_nparticle[z,:,1,0], k_nearest_coord_nparticle[z,:,1,1]]
        predict_down_left = U[z, :, k_nearest_coord_nparticle[z,:,2,0], k_nearest_coord_nparticle[z,:,2,1]]
        predict_down_right = U[z, :, k_nearest_coord_nparticle[z,:,3,0], k_nearest_coord_nparticle[z,:,3,1]]

        # print(predict_up_left.shape, predict_up_right.shape, predict_down_left.shape, predict_down_right.shape)
        # print(predict_up_left, predict_up_right, predict_down_left, predict_down_right)
        predict_cat = torch.stack((predict_up_left, predict_up_right, predict_down_left, predict_down_right), dim=-1)  # shape = DIM x nparticles x k_nearest
        # print('predict_cat', predict_cat.shape)
        predict_app.append(predict_cat)

    predict_k_nearest_force = torch.stack(predict_app)  # sample x  DIM   x npartilce x k_nearest

    print('predict_k_nearest_force')
    print(predict_k_nearest_force)

    z_l = torch.sum(1. / k_nearest_distance_nparticle, dim=-1)   #nsample x nparticle
    z_l = z_l.unsqueeze(dim=1) #nsample x 1 x nparticle
    print('z_l', z_l)

    k_nearest_distance_nparticle_unsqueeze = k_nearest_distance_nparticle.unsqueeze(dim = 1) #nsample x 1 x nparticle x k_nearest
    print('k_nearest_distance_nparticle')
    print(k_nearest_distance_nparticle_unsqueeze)
    print(k_nearest_distance_nparticle_unsqueeze.shape)

    predict_each_particle = 1./z_l * (torch.sum(1./k_nearest_distance_nparticle_unsqueeze * predict_k_nearest_force, dim=-1)) #nsample x DIM x nparticle
    predict_each_particle = predict_each_particle.permute((0,2,1)) #nsample x nparticle x DIM
    print('predict each particle')
    print(predict_each_particle)

    if (k_nearest_distance_nparticle_unsqueeze < 0.001).any() :
        index = torch.where(k_nearest_distance_nparticle < 0.001) #nsample x nparticle x k_nearest
        print(index)
        predict_k_vv_nearest_force = predict_k_nearest_force[index[0],:,index[1], index[2]]   # sample x  DIM   x npartilce x k_nearest
        print(predict_k_vv_nearest_force)
        predict_each_particle[index[0],index[1]] = predict_k_vv_nearest_force.double()

    print('predict each particle')
    print(predict_each_particle)
    print(predict_each_particle.shape)