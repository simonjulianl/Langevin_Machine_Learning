import HNNwLJ.hamiltonian as noML_hamiltonian
from HNNwLJ.hamiltonian.LJ_term import LJ_term
from HNNwLJ.hamiltonian.lennard_jones import lennard_jones
from HNNwLJ.hamiltonian.kinetic_energy import kinetic_energy
import torch
from HNNwLJ.hamiltonian.pb import pb
from HNNwLJ.phase_space.phase_space import phase_space
from HNNwLJ.pair_wise_HNN.dataset_split import dataset_split
from HNNwLJ.integrator import linear_integrator
from HNNwLJ.integrator.methods import linear_velocity_verlet
from HNNwLJ20210128.parameters.ML_paramaters import ML_parameters
from HNNwLJ.pair_wise_HNN import pair_wise_HNN
from HNNwLJ.pair_wise_HNN.models import pair_wise_MLP
from HNNwLJ.pair_wise_HNN.loss import qp_MSE_loss
import torch.optim as optim

if __name__ == '__main__':

    mass = 1
    epsilon = 1.
    sigma = 1.
    boxsize = 6.
    DIM = 2
    nsamples = 1
    npixels = 16

    nsamples_label = 10
    nsamples_ML = 1
    nparticle = 2
    iterations = 10 # 10 steps to pair with large time step = 0.1
    tau_long = 0.1 # short time step for label
    tau_short = 0.01

    noML_hamiltonian = noML_hamiltonian.hamiltonian()
    lj_term = LJ_term(epsilon=epsilon,sigma=sigma,boxsize=boxsize)
    noML_hamiltonian.append(lennard_jones(lj_term, boxsize))
    noML_hamiltonian.append(kinetic_energy(mass))

    phase_space = phase_space()
    pb = pb()

    state = {

        'nsamples_cur': 0,
        'nsamples_label': nsamples_label,
        'nsamples_ML': nsamples_ML,
        'nparticle' : nparticle,
        'npixels' : npixels,
        'epsilon' : 1.,
        'sigma' : 1.,
        'DIM' : DIM,
        'boxsize' : boxsize,
        'iterations' : iterations,
        'MD_iterations': 0,
        'tau_cur': 0.0,
        'tau_long': tau_long,  # for MD
        'tau_short': tau_short,  # for label (gold standard)
        'integrator_method' : linear_velocity_verlet,
        'phase_space' : phase_space,
        'pb_q' : pb

        }

    filename = '../init_config/N_particle{}_samples{}_rho0.1_T0.04_pos_sampled.pt'.format(nparticle, nsamples_label)
    dataset = dataset_split(filename, **state)
    train_data, valid_data = dataset.hamiltonian_dataset(ratio=0.8)
    # train_label = dataset.phase_space2label(train_data, linear_integrator, noML_hamiltonian)
    valid_q_label, valid_p_label = dataset.phase_space2label(valid_data, linear_integrator, noML_hamiltonian)
    print(valid_q_label[-1],valid_p_label[-1])

