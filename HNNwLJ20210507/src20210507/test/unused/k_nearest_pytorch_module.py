import torch
import torch.nn.functional as F
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

def euclidean_distance(vector1, vector2):
    # print('distance', vector1, vector2)
    # print('distance', vector1.shape, vector2.shape)
    return torch.sqrt(torch.sum(torch.pow(vector1 - vector2, 2), dim=1))  # sum DIM that is dim=1 <- nparticle x DIM

def k_nearest_grids( q_list, grid_list):
    # to calculate distance between particle and grid
    print(q_list.shape)
    # q_list shape is [ nsamples, DIM=(x,y) ]
    # grid_list shape is [ gridL * gridL , DIM = (x,y)]
    q_list = q_list[:,0] # one particle
    print('q list', q_list)
    print('grid list', grid_list.shape)
    grid_list0 = torch.unsqueeze(grid_list, dim=1)
    grid_list = torch.repeat_interleave(grid_list0, q_list.shape[0], dim=1)
    print('grid list', grid_list.shape)
    dist = torch.norm(q_list - grid_list, dim = (1,2))
    print(dist)
    print(dist.shape)
    knn = dist.topk(4, largest=False)
    print('knn', knn)
    quit()
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

    nsamples = 1
    nparticle = 2
    npixels = 8
    boxsize = 2
    DIM = 2
    grid_interval = boxsize / npixels
    print('grid interval', grid_interval)

    grid = build_gridpoint(npixels, boxsize, DIM)
    # print('grid', grid)
    U = torch.rand(2, npixels, npixels)
    print('U', U)
    # print(U.shape)

    # U = torch.rand(1, 2, npixels, npixels)
    # print('U', U)
    # print(U.shape)

    # q_list = [[-0.5, -0.5],[0.5, 0.5]]
    # q_list = [[[-0.95, -0.95],[0.5, 0.5]], [[-0.6001, -0.6],[0.05, 0.3]], [[-0.63, -0.2],[0.4, 0.2]]]
    q_list = [[[-0.95, -0.95],[0.5, 0.5]]]

    _q_list = torch.tensor(q_list, dtype=torch.float64)

    k_nearest_nparticle_app = []
    k_nearest_coord_nparticle_app = []

    # Distance btw each particle and 4 nearest grids
    # 4 nearest girds coordinates

    # print(_q_list[:,i].shape)
    # print(_q_list[:,i])  # nsamples x each particle

    k_nearest = k_nearest_grids(_q_list, grid)
    print('k nearest', k_nearest)
    # print(k_nearest.shape)

