import os
import torch

class checkpoint:
    """
    this class to save and load checkpoints in a dictionary
    """
    def __init__(self, check_path, any_netA, any_netB, opt):
        """
        check_path  : path to save or load checkpoints
        any_netA    : first NN for pairwise or optical flow
        any_netB    : second NN for pairwise or optical flow
        opt         : pass optimizer
        """

        self.check_path = check_path
        self.any_netA = any_netA
        self.any_netB = any_netB
        self._opt = opt

    # ===================================================
    def load_checkpoint(self, load_filename):

        ''' function to load saved model
            remember to first initialize the model and optimizer, then load the dictionary

        Parameters
        ----------
        load_filename : string
                load saved model. if not, quit
        '''

        full_name = self.check_path + load_filename

        if os.path.isfile(full_name):

            print("=> loading checkpoint '{}'".format(full_name))
            checkpoint = torch.load(full_name)[0]

            # load models weights state_dict
            self.any_netA.load_state_dict(checkpoint['modelA_state_dict'])
            self.any_netB.load_state_dict(checkpoint['modelB_state_dict'])
            print('Previously trained modelA weights state_dict loaded...')
            print('Previously trained modelB weights state_dict loaded...')

            self._opt.load_state_dict(checkpoint['optimizer'])
            print('Previously trained optimizer state_dict loaded...')

        else:
            print("=> no checkpoint found at '{}'".format(full_name))
            quit()

    # ===================================================
    def save_checkpoint(self, save_filename):

        ''' function to record the state after each training

        Parameters
        ----------
        save_filename : string
            path to the saving the checkpoint
        '''

        full_name = self.check_path + save_filename
        torch.save({'modelA_state_dict' : self.any_netA.state_dict(),
                    'modelB_state_dict' : self.any_netB.state_dict(),
                    'optimizer' : self._opt.state_dict()}, full_name)


