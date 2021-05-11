import torch

class get_paired_distance_indices:

    ''' get_paired_distance_indices class to do fast extraction of indices
    staticmethod is callable without instantiating the class first

    use in paired_distance_reduced of pb class and delta_qorp of pair_wise_HNN class
    '''

    permute_order = (0,2,1,3) # need to calculate paired_distance

    @staticmethod
    def get_indices(s):

        ''' function to obtain indices of non-zero values that do not consider interactions
            btw themself ex) q_1x, q_1x or q_1y, q_1y or q_2x, q_2x ....

        parameters
        ----------
        s : torch.tensor
                dq list shape
                shape is [nsamples, nparticle, nparticle, DIM]
		DIM = 2 for 2D LJ model
        n : nparticle
        m : torch.tensor
                make 1 all of s shape and then become 0 when consider interactions btw themself
        Returns
        ----------
        indices of non-zero values in m
        '''

        n = s[1]
        m = torch.ones(s)

        # check - run "python3.7 -O" to switch off
        #if __debug__: 
        #    m2 = torch.clone(m)
        #    for i in range(n): m2[:,i,i,:] = 0

        dm = torch.diagonal(m,0,1,2) # offset, nparticle, nparticle
        torch.fill_(dm,0.0)

        #if __debug__:
        #    err = (m-m2)**2
        #    assert (torch.sum(err)<1e-6),'error in diagonal computations'

        return m.nonzero(as_tuple=True)

    @staticmethod
    def reduce(delta, indices):

        ''' function to obtain delta of non-zero indices that do not consider themself interactions

        parameters
        ----------
        delta : torch.tensor
                pass shape is [nsamples, nparticle, nparticle, DIM]
                distances between particle interactions ex) q_1x - q_1x, q_1x - q_2x, ..., q_ny - q_ny
        indices :
                is return of get_indices

        Returns
        ----------
        shape is [nsamples, nparticle, (nparticle - 1), DIM ]
        '''
        return delta[indices]
