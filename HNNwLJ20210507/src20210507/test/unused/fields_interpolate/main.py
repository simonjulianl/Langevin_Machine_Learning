import torch
from interpolator import interpolator
from phase_space import phase_space
import plot_grids

if __name__ == "__main__":
    ''' gird range is [-0.5 * boxsize, 0.5 * boxsize] 
        so that shift gridL range as [ 0, x or y / grid_interval + boxsize] '''

    seed = 9372211
    torch.manual_seed(seed)

    gridL = 8
    boxsize = 4.

    interpolator =  interpolator()
    phase_space  =  phase_space()

    # q_list = torch.tensor([[[-1.3, -1.9], [-0.9,1.2],[0.9,0.2]], [[-1.3, -1.9], [-0.9,1.2],[0.6,1.2]]])
    q_list = torch.tensor([[[-1.9, -1.3], [1.2, -0.9]], [[-1.9, -1.3], [1.2, -0.9]]])
    print('q list', q_list)

    phase_space.set_q(q_list)
    phase_space.set_boxsize(boxsize)

    grids_list = plot_grids.build_grid_list(phase_space, gridL)
    plot_grids.show_grids_nparticles(grids_list, q_list)

    U = torch.rand(2, 2, gridL, gridL)  # e.g forces on grids ( gridL * gridL )
    # U.shape is [nsamples, DIM=(fx,fy), gridLx, gridLy]
    print('phi predict', U)

    interpolate = interpolator.nsamples_interpolator(U, phase_space)
    # interpolate.shape is [nsampcles, nparticle, DIM=(x,y)]
    print(interpolate)
