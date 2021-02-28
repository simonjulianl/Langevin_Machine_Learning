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

if __name__ == '__main__':

    nparticle = 2
    npixels = 10
    boxsize = 2
    DIM = 2
    grid_interval = boxsize / npixels
    grid = build_gridpoint(npixels, boxsize, DIM)

    # q_list = [[-0.5, -0.5],[0.5, 0.5]]
    q_list = [[-0.95, -0.95],[0.5, 0.5]]

    _q_list = torch.tensor(q_list, dtype=torch.float64)
    print('q_list', _q_list)
    # show_grid_nparticles(grid, _q_list)
    for i in range(nparticle):
        print(i)
        k_nearest_up_left = torch.floor(_q_list[i] / grid_interval ) * grid_interval
        k_nearest_up_left_x = k_nearest_up_left[0]
        k_nearest_up_left_y = k_nearest_up_left[1]

        k_nearest_up_right = torch.tensor([k_nearest_up_left_x, k_nearest_up_left_y + grid_interval])
        k_nearest_down_left = torch.tensor([k_nearest_up_left_x + grid_interval, k_nearest_up_left_y])
        k_nearest_down_right = torch.tensor([k_nearest_up_left_x + grid_interval, k_nearest_up_left_y + grid_interval])

        print(k_nearest_up_left)
        print(k_nearest_up_right)
        print(k_nearest_down_left)
        print(k_nearest_down_right)

        up_left_coord = ( boxsize / 2 + k_nearest_up_left ) / grid_interval
        up_left_index = up_left_coord[0] * npixels + up_left_coord[1]

        up_right_coord = (boxsize / 2 + k_nearest_up_right) / grid_interval
        up_right_index = up_right_coord[0] * npixels + up_right_coord[1]

        down_left_coord = (boxsize / 2 + k_nearest_down_left) / grid_interval
        down_left_index = down_left_coord[0] * npixels + down_left_coord[1]

        down_right_coord = (boxsize / 2 + k_nearest_down_right) / grid_interval
        down_right_index = down_right_coord[0] * npixels + down_right_coord[1]

        print(up_left_coord, up_right_coord, down_left_coord, down_right_coord)
        print(up_left_index, up_right_index, down_left_index, down_right_index)