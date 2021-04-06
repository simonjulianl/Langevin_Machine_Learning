import torch
import itertools
from parameters.ML_parameters           import ML_parameters
from hamiltonian.hamiltonian            import hamiltonian
from hamiltonian.lennard_jones          import lennard_jones
from hamiltonian.kinetic_energy         import kinetic_energy
from utils.get_paired_distance_indices  import get_paired_distance_indices


class pairwise_HNN(hamiltonian):

    ''' pairwise_HNN class to learn dHdq and then combine with nodHdq '''

    _obj_count = 0

    def __init__(self, pairwise_MLP1, pairwise_MLP2, tau_cur):

        pairwise_HNN._obj_count += 1
        assert (pairwise_HNN._obj_count == 1),type(self).__name__ + " has more than one object"

        super().__init__()
        self.net1 = pairwise_MLP1
        self.net2 = pairwise_MLP2

        # # append term to calculate dHdq
        super().append(lennard_jones())
        super().append(kinetic_energy())

        self.tau_cur = tau_cur

    # ===================================================
    def net_parameters(self):
        return itertools.chain(self.net1.parameters(), self.net2.parameters())
        #return list(self.net1.parameters())+list(self.net2.parameters())
    # ===================================================
    def train(self):
        self.net1.train()  # pytorch network for training
        self.net2.train()  # pytorch network for training

    # ===================================================
    def eval(self):   # pytorch network for eval
        self.net1.eval()
        self.net2.eval()

    # ===================================================
    def dHdq1(self, phase_space):
        return self.dHdq_all(phase_space,self.net1)
    # ===================================================
    def dHdq2(self, phase_space):
        return self.dHdq_all(phase_space,self.net2)
    # ===================================================
    def dHdq_all(self, phase_space, net):

        ''' function to calculate dHdq = noML_dHdq + residual ML_dHdq

        Parameters
        ----------
        q_list : torch.tensor
                shape is [(nsamples, nparticle, DIM]
        noML_dHdq : torch.tensor (cpu)
                shape is [nparticle, DIM]
                use .to(device) to move a tensor to device
        predict : torch.tensor (gpu)
                shape is [nparticle, DIM]
        Returns
        ----------
        corrected_dHdq : torch.tensor
                shape is [nparticle, DIM]

        '''
        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        nsamples, nparticle, DIM = q_list.shape

        noML_dHdq = super().dHdq(phase_space)
        noML_dHdq = noML_dHdq.squeeze(dim=0)

        x = self.pack_dqdp_tau(phase_space, self.tau_cur)
        # x.shape = [nsamples * nparticle * nparticle, 5]

        predict = net(x)
        predict = self.unpack_dqdp_tau(predict,q_list.shape)
        # corrected_dHdq.shape = [nsamples,nparticle,DIM]
        corrected_dHdq = noML_dHdq + predict

        return corrected_dHdq

    # ===================================================
    def pack_dqdp_tau(self, phase_space, tau_cur):

        ''' function to prepare input in nn
        this function is use to make delta_q,delta_p,tau for input into model

        Parameters
        ----------
        tau_cur : float
                large time step for input in neural network
        phase_space : contains q_list, p_list as input for integration

        Returns
        ----------
        input in neural network
        shape is [(nsamples x nparticle x nparticle-1), (del_qx, del_qy, del_px, del_py, tau )]

        '''

        nsamples, nparticle, DIM = phase_space.get_q().shape

        dqdp_list = self.make_dqdp(phase_space)
        # dqdp_list.shape = [nsamples,nparticle,nparticle,2*DIM]

        tau_tensor = torch.zeros([nsamples,nparticle,nparticle,1],dtype=torch.float64)
        tau_tensor.fill_(tau_cur)
        x = torch.cat((dqdp_list,tau_tensor),dim = 3)
        x = torch.reshape(x,(nsamples*nparticle*nparticle,5))
        return x

    # ===================================================

    def unpack_dqdp_tau(self,y,qlist_shape):
        # y.shape = [nsamples*nparticle*nparticle,2]

        nsamples, nparticle, DIM = qlist_shape

        y = torch.reshape(y,(nsamples,nparticle,nparticle,DIM))

        # check - run "python3.7 -O" to switch off
        if __debug__:
            y2 = torch.clone(y)
            for i in range(nparticle): y2[:,i,i,:] = 0

        dy = torch.diagonal(y,0,1,2) # offset,nparticle,nparticle
        torch.fill_(dy,0.0)

        if __debug__:
            err = (y-y2)**2
            assert (torch.sum(err)<1e-6),'error in diagonal computations'

        # SJ check is sum over dim=2 is correct
        # y.shape = [nsamples,nparticle,nparticle,2]
        y = torch.sum(y,dim=2) # check if this is correct?? -- need to sum over all neighbors

        return y

    # ===================================================

    def make_dqdp(self, phase_space):

        ''' function to make dq and dp for feeding into nn

        Returns
        ----------
        take q_list and p_list, generate the difference matrix
        q_list.shape = p_list.shape = [nsamples,nparticle,DIM]
        dqdp : shape is [ nsamples, nparticle, nparticle, 4]
        here 4 is (dq_x, dq_y, dp_x, dp_y )
        '''

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        dq = self.delta_state(q_list)
        dp = self.delta_state(p_list)

        dqdp = torch.cat((dq, dp), dim=-1)

        return dqdp

    # ===================================================

    def delta_state(self, state_list):

        ''' function to calculate distance of q or p between two particles

        Parameters
        ----------
        state_list : torch.tensor
                shape is [nsamples, nparticle, DIM]
        state_len : nparticle
        state0 : shape is [nsamples, 1, nparticle, DIM]
                new tensor with a dimension of size one inserted at the specified position (dim=1)
        statem : shape is [nsamples, nparticle, nparticle, DIM]
                repeated tensor which has the same shape as state0 along with dim=1
        statet : shape is [nsamples, nparticle, nparticle, DIM]
                permutes the order of the axes of a tensor

        Returns
        ----------
        dstate : shape is [nsamples, nparticle, nparticle, DIM]
                distance of q or p btw two particles
        '''

        state_len = state_list.shape[1]
        state0 = torch.unsqueeze(state_list, dim=1)
        statem = torch.repeat_interleave(state0, state_len, dim=1)

        statet = statem.permute(get_paired_distance_indices.permute_order)

        dstate = statet - statem

        return dstate

