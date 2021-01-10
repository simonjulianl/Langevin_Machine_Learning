# train model
import hamiltonian as NoML_hamiltonian
from LJ_term import LJ_term
from lennard_jones import lennard_jones
from kinetic_energy import kinetic_energy
import torch
from pb import pb
from phase_space import phase_space
from linear_integrator import linear_integrator
from linear_velocity_verlet import linear_velocity_verlet
from pair_wise_HNN_test import pair_wise_HNN
from pair_wise_MLP import pair_wise_MLP

if __name__ == '__main__':

    q_list = [[[3,2],[2.2,1.21]]]
    p_list = [[[0.,0.],[0.,0.]]]
    # q_list = [[[2.3945972, 0.79560974], [1.29235072, 0.64889931], [1.66907468, 1.693532]]]
    # p_list = [[[0.1,0.],[0.,0.4],[0.1, 0.3]]]

    q_list_tensor, p_list_tensor = torch.tensor([q_list,p_list])

    pb = pb()
    phase_space = phase_space()

    nsample = 1
    N_particle = 2
    DIM = 2
    mass = 1
    epsilon = 1.
    sigma = 1.
    boxsize = 6.
    tau = 0.1
    iterations = 1
    n_input = 5
    n_hidden = 10

    NoML_hamiltonian = NoML_hamiltonian.hamiltonian()
    lj_term = LJ_term(epsilon=epsilon,sigma=sigma,boxsize=boxsize)
    NoML_hamiltonian.append(lennard_jones(lj_term, boxsize))
    NoML_hamiltonian.append(kinetic_energy(mass))

    nepochs = 10

    state = {
        'hamiltonian' : NoML_hamiltonian,
        'N' : nsample,
        'particle' : N_particle,
        'DIM' : DIM,
        'BoxSize' : boxsize,
        'iterations' : iterations,
        'tau' : tau,
        'integrator_method' : linear_velocity_verlet,
        'phase_space' : phase_space,
        'pb_q' : pb

        }

    MLP = pair_wise_MLP(n_input,n_hidden)
    pair_wise_HNN = pair_wise_HNN(MLP, **state) # data preparation / calc f_MD, f_ML
    # print(pair_wise_HNN.network(n_input,n_hidden) )
    # print(pair_wise_HNN.noML_hamiltonian)

    pair_wise_HNN.train()

    state['phase_space'].set_q(q_list_tensor)
    state['phase_space'].set_p(p_list_tensor)

    #pair_wise_HNN.phase_space2data(state['phase_space'],pb)
    #print(pair_wise_HNN.dHdq(state['phase_space'],pb))
    MD_integrator = linear_integrator(**state)

    for e in range(nepochs):

        MD_integrator.integrate()

