from parameters.MD_parameters        import MD_parameters
from parameters.ML_parameters        import ML_parameters
from HNN.data_handler              import data_handler
from phase_space                     import phase_space
from integrator.linear_integrator    import linear_integrator

import torch

if __name__=='__main__':

    tau_long         = MD_parameters.tau_long
    tau_cur          = tau_long

    # io varaiables
    dataset = data_handler()
    phase_space = phase_space.phase_space()
    hamiltonian_obj = MD_parameters.hamiltonian_obj
    linear_integrator_obj = linear_integrator( MD_parameters.integrator_method, MD_parameters.integrator_method_backward )
 
    torch.manual_seed(2233)

    # change to read from file (in this file, large time step 0.1 / short time step 0.001)
    dataset.load(ML_parameters.train_data_file)
    qp_input_train, qp_label_train = dataset._shuffle(dataset.qp_list_input, dataset.qp_list_label)

    print('input shape', qp_input_train.shape)
    print('label shape', qp_label_train.shape)
    print('tau long', tau_long, 'tau short', dataset.tau_short)

    assert (abs(tau_long - dataset.tau_long)<1e-6), 'incorrect tau long value'

    nepoch = 20000

    optimizer = ML_parameters.opt.create(hamiltonian_obj.net_parameters())

    for e in range(nepoch):

        phase_space.set_q(qp_input_train[0])
        phase_space.set_p(qp_input_train[1])
        qp_list = linear_integrator_obj.one_step(hamiltonian_obj, phase_space, tau_cur)

        # qp_list.shape = [2,nsamples,nparticle,DIM]

        q_out = qp_list[0]
        p_out = qp_list[1]

        loss = torch.mean((q_out - qp_label_train[0])**2 + (p_out - qp_label_train[1])**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if e%(nepoch//100)==0: print('e ',e,' loss ',loss)

