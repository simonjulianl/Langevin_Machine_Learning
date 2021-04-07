import os
import torch

class checkpoint:
    """
    this class to save and load checkpoints in a dictionary
    """
    def __init__(self, check_path, any_netA, any_netB, optA, optB):
        """
        check_path  : path to save or load checkpoints
        any_net1    : first NN for pairwise or optical flow
        any_net2    : second NN for pairwise or optical flow
        optA         : pass optimizerA
        optB         : pass optimizerB
        """

        self.check_path = check_path
        self.any_netA = any_netA
        self.any_netB = any_netB
        self._optA = optA
        self._optB = optB

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

            self._optA.load_state_dict(checkpoint['optimizerA'])
            self._optB.load_state_dict(checkpoint['optimizerB'])
            print('Previously trained optimizerA state_dict loaded...')
            print('Previously trained optimizerB state_dict loaded...')


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
                    'optimizerA' : self._optA.state_dict(),
                    'optimizerB' : self._optB.state_dict()}, full_name)


