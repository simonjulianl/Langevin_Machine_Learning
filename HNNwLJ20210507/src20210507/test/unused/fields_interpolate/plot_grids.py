import torch
import matplotlib.pyplot as plt

def build_grid_list(phase_space, gridL):
    '''
    Parameters
    ----------
    grids_list : torch.tensor
            shape is [gridL * gridL, DIM=(x coord, y coord)]
    '''

    nsamples, nparticle, DIM = phase_space.get_q().shape
    boxsize = phase_space.get_boxsize()

    xvalues = torch.arange(0, gridL, dtype=torch.float64)
    xvalues = xvalues - 0.5 * gridL
    yvalues = torch.arange(0, gridL, dtype=torch.float64)
    yvalues = yvalues - 0.5 * gridL

    # create grids list shape is [gridL * gridL, 2]
    gridx, gridy = torch.meshgrid(xvalues * (boxsize / gridL), yvalues * (boxsize / gridL))
    grids_list = torch.stack([gridx, gridy], dim=-1)
    grids_list = grids_list.reshape((-1, DIM))

    return grids_list

def show_grids_nparticles(grids_list, q_list):

    for i in range(2): # show two samples

        plt.title('sample {}'.format(i))
        plt.plot(grids_list[:,0], grids_list[:,1], marker='.', color='k', linestyle='none', markersize=12)
        plt.plot(q_list[i,:, 0], q_list[i,:, 1], marker='x', color='r', linestyle='none', markersize=12)
        plt.show()
        plt.close()