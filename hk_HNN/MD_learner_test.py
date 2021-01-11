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
from pair_wise_HNN import pair_wise_HNN
from pair_wise_MLP import pair_wise_MLP
from loss import qp_MSE_loss
import torch.optim as optim


def phase_spacedata(MD_integrator, noML_hamiltonian, **state):

    state = MD_integrator.integrate(noML_hamiltonian)

    q_list_label = phase_space.get_q()
    p_list_label = phase_space.get_p()

    return (q_list_label, p_list_label)


if __name__ == '__main__':

    seed = 9372211
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the model functions nondeterministically.
    # torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

    q_list = [[[3,2],[2.2,1.21]]]
    p_list = [[[0.1,0.1],[0.,0.]]]
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
    n_hidden = 5
    lr = 0.01


    NoML_hamiltonian = NoML_hamiltonian.hamiltonian()
    lj_term = LJ_term(epsilon=epsilon,sigma=sigma,boxsize=boxsize)
    NoML_hamiltonian.append(lennard_jones(lj_term, boxsize))
    NoML_hamiltonian.append(kinetic_energy(mass))

    nepochs = 1

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

    MLP = pair_wise_MLP(n_input,n_hidden)
    print('model',MLP)
    pairwise_HNN = pair_wise_HNN(NoML_hamiltonian, MLP, **state) # data preparation / calc f_MD, f_ML
    print('pairwise_HNN',pairwise_HNN)
    # print(pair_wise_HNN.network(n_input,n_hidden) )
    # print(pair_wise_HNN.noML_hamiltonian)

    pairwise_HNN.train()
    opt = optim.Adam(MLP.parameters(), lr=lr)

    state['phase_space'].set_q(q_list_tensor)
    state['phase_space'].set_p(p_list_tensor)

    # pair_wise_HNN.phase_space2data(state['phase_space'],pb)
    # print(pair_wise_HNN.dHdq(state['phase_space'],pb))
    MD_integrator = linear_integrator(**state)

    label = phase_spacedata( MD_integrator, NoML_hamiltonian, **state)
    print('label', label)


    torch.autograd.set_detect_anomaly(True) # get the method causing the NANs

    for e in range(nepochs):

        # q_list_tensor = q_list_tensor
        # p_list_tensor = p_list_tensor

        state['phase_space'].set_q(q_list_tensor)
        state['phase_space'].set_p(p_list_tensor)

        q_list_predict_, p_list_predict_ = MD_integrator.integrate(pairwise_HNN)  # general hamiltonain ( MD + ML)
        q_list_predict = q_list_predict_.reshape(-1, q_list_predict_.shape[2], q_list_predict_.shape[3])
        p_list_predict = p_list_predict_.reshape(-1, p_list_predict_.shape[2], p_list_predict_.shape[3])

        prediction = (q_list_predict, p_list_predict)

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
    prediction_noML = phase_spacedata(MD_integrator, NoML_hamiltonian, **state )


    print('prediction with   ML', prediction)
    print('prediction with noML', prediction_noML)
    q_pred, p_pred = prediction
    q_label, p_label = prediction_noML
    now_loss = (q_pred - q_label)**2  + (p_pred - p_label)**2
    now_loss = torch.sum(now_loss)
    train_loss = qp_MSE_loss(prediction, label)
    print('previous loss', train_loss)
    print('now      loss', now_loss)

