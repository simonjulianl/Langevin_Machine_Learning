from pb import pb

class phase_space(pb):

    ''' phase space class that have q, p, and boxsize
        copy q_list to _q_list and p_list to _p_list
    '''

    _obj_count = 0

    def __init__(self):

        super().__init__()

        phase_space._obj_count += 1
        assert(phase_space._obj_count <= 2), type(self).__name__ + ' has more than two objects'
        # one phase space object for the whole code
        # the other phase space object only use as a copy in lennard-jones class in dimensionless
        # form

        '''initialize phase space of [nsamples, nparticle, DIM] '''
        self._q_list = None
        self._p_list = None
        self._boxsize = None
        print('phase_space initialized')

    def set_p(self, p_list):
        self._p_list = p_list.clone()

    def set_q(self, q_list):
        self._q_list = q_list.clone()

    def set_boxsize(self,boxsize):
        self._boxsize = boxsize

    def get_p(self):
        return self._p_list.clone()

    def get_q(self):
        return self._q_list.clone()

    def get_boxsize(self):
        return self._boxsize
