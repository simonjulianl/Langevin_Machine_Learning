import os
import torch

class checkpoint:
    """
    this class to save and load checkpoints in a dictionary
    """
    def __init__(self, net_list, opt = None, sch = None):
        """
        net_list    : network list for chk pt
        opt         : pass optimizer
        """

        self.net_list = net_list
        self._opt = opt
        self._sch = sch
        print('checkpoint initialized : net list ',net_list,' opt ',opt, 'sch', sch)

    # ===================================================
    def load_checkpoint(self, load_filename):

        ''' function to load saved model
            remember to first initialize the model and optimizer, then load the dictionary

        Parameters
        ----------
        load_filename : string
                load saved model. if not, quit
        '''

        full_name = load_filename

        if os.path.isfile(full_name):

            print("=> loading checkpoint '{}'".format(full_name))
            checkpoint = torch.load(full_name)
            print(checkpoint)

            # load models weights state_dict
            for net_id, nlist in enumerate(checkpoint['net_list']):
                self.net_list[net_id].load_state_dict(nlist.state_dict())
                print('Previously net_list state_dict loaded...')

            if self._opt is not None:
                self._opt.load_state_dict(checkpoint['optimizer'])
                print('Previously trained optimizer state_dict loaded...')

            if self._sch is not None:
                self._sch.load_state_dict(checkpoint['scheduler'])
                print('Previously trained scheduler state_dict loaded...')

        else:
            print("=> no checkpoint found at '{}'".format(full_name))
            # quit() HK : comment for MD sampler when run noML

    # ===================================================
    def save_checkpoint(self, save_filename):

        ''' function to record the state after each training

        Parameters
        ----------
        save_filename : string
        '''

        assert (self._opt or self._sch is not None), 'opt or sch is None ... so that save error ...'

        full_name = save_filename
        torch.save({'net_list' : self.net_list,
                    'optimizer' : self._opt.state_dict(),
                    'scheduler' : self._sch.state_dict()}, full_name)


