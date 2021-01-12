import HNNwLJ.hamiltonian as NoML_hamiltonian
from HNNwLJ.hamiltonian.LJ_term import LJ_term
from HNNwLJ.hamiltonian.lennard_jones import lennard_jones
from HNNwLJ.hamiltonian.kinetic_energy import kinetic_energy
import torch
from HNNwLJ.hamiltonian.pb import pb
from HNNwLJ.phase_space.phase_space import phase_space
from HNNwLJ.integrator import linear_integrator
from HNNwLJ.integrator.methods import linear_velocity_verlet
from HNNwLJ.pair_wise_HNN import pair_wise_HNN
from HNNwLJ.pair_wise_HNN.models import pair_wise_MLP
from HNNwLJ.pair_wise_HNN.loss import qp_MSE_loss
import torch.optim as optim


def phase_space2label(MD_integrator, noML_hamiltonian):

    label = MD_integrator.integrate(noML_hamiltonian)

    return label


if __name__ == '__main__':

    seed = 9372211
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the model functions nondeterministically.
    # torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    # q_list = [[[3,2],[2.2,1.21]]]
    # p_list = [[[0.,0.],[0.,0.]]]
    q_list = [[[-0.95692938,  1.87027837],[-2.08708374,  2.03361165]]]
    p_list = [[[ 0.08764001, -0.20370242],[-0.05110963, -0.01624577]]]
    #
    # q_list = [[[2.3945972, 0.79560974], [1.29235072, 0.64889931], [1.66907468, 1.693532]]]
    # p_list = [[[0.2,0.],[0.,0.4],[0.1, 0.1]]]

    q_list_tensor, p_list_tensor = torch.tensor([q_list,p_list])

    pb = pb()
    phase_space = phase_space()

    # q_list_tensor, p_list_tensor = phase_space.read('N_particle2_samples1_rho0.1_T0.04_pos_sampled.npy',nsamples=1)
    # print(q_list_tensor, p_list_tensor)


    nsample = 1
    N_particle = 2
    DIM = 2
    mass = 1
    epsilon = 1.
    sigma = 1.
    boxsize = 6.
    tau = 0.01
    iterations = 10
    n_input = 5
    n_hidden = 5
    lr = 0.01
    nepochs = 1

    NoML_hamiltonian = NoML_hamiltonian.hamiltonian()
    lj_term = LJ_term(epsilon=epsilon,sigma=sigma,boxsize=boxsize)
    NoML_hamiltonian.append(lennard_jones(lj_term, boxsize))
    NoML_hamiltonian.append(kinetic_energy(mass))

    state = {
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

    #=== prepare label ====================================================#
    state['phase_space'].set_q(q_list_tensor)
    state['phase_space'].set_p(p_list_tensor)

    label = phase_space2label(linear_integrator(**state), NoML_hamiltonian)
    print('label', label)
    #===== end ============================================================#

    torch.autograd.set_detect_anomaly(True) # get the method causing the NANs

    # to prepare data at large time step, need to change tau and iterations
    # tau = large time step 0.1 and 1 step
    state['tau'] = state['tau'] * state['iterations']   # tau = 0.1
    state['iterations'] = int(state['tau'] * state['iterations']) # 1 step

    MLP = pair_wise_MLP(n_input,n_hidden)
    pairwise_HNN = pair_wise_HNN(NoML_hamiltonian, MLP, **state) # data preparation / calc f_MD, f_ML

    # print(pair_wise_HNN.network(n_input,n_hidden) )
    # print(pair_wise_HNN.noML_hamiltonian)

    pairwise_HNN.train()
    opt = optim.Adam(MLP.parameters(), lr=lr)

    # pair_wise_HNN.phase_space2data(state['phase_space'],pb)
    # print(pair_wise_HNN.dHdq(state['phase_space'],pb))

    for e in range(nepochs):

        state['phase_space'].set_q(q_list_tensor)
        state['phase_space'].set_p(p_list_tensor)

        prediction = linear_integrator(**state).integrate(pairwise_HNN)  # general hamiltonain ( MD + ML)

        print(prediction)

        loss = qp_MSE_loss(prediction, label)

        print('loss',loss)
        opt.zero_grad()  # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
        loss.backward()  # backward pass : compute gradient of the loss wrt model parameters
        train_loss = loss.item()  # get the scalar output
        opt.step()


        print('w 1 grad',MLP.correction_term[0].weight.grad)
        print('w 2 grad', MLP.correction_term[2].weight.grad)


        print('epoch loss ',e,train_loss)

    # do one step velocity verlet without ML
    print('\n')
    print('do one step velocity verlet without ML')
    print(state)

    state['phase_space'].set_q(q_list_tensor)
    state['phase_space'].set_p(p_list_tensor)

    prediction_noML = phase_space2label(linear_integrator(**state), NoML_hamiltonian ) # here label at large time step

    print('prediction with   ML', prediction)
    print('prediction with noML', prediction_noML)

    q_pred,  p_pred  = prediction
    q_label, p_label = prediction_noML

    now_loss = (q_pred - q_label)**2  + (p_pred - p_label)**2
    now_loss = torch.sum(now_loss)
    train_loss = qp_MSE_loss(prediction, label)
    print('previous loss', train_loss)  # label at short time step 0.01
    print('now      loss', now_loss)   # label at large time step 0.1

