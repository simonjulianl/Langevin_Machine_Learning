import os
import torch
import shutil

class checkpoint:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, check_path, any_network, opt):
        """
        add things here.....
        """
        self.any_network = any_network
        self._opt = opt
        self.check_path = check_path

    def load_checkpoint(self, load_filename):

        ''' function to load saved model

        Parameters
        ----------
        load_filename : string
                load saved model. if not, pass
        '''

        full_name = self.check_path + load_filename

        if os.path.isfile(full_name):

            print("=> loading checkpoint '{}'".format(full_name))
            checkpoint = torch.load(full_name)[0]

            # load models weights state_dict
            self.any_network.load_state_dict(checkpoint['model_state_dict'])

            print('Previously trained models weights state_dict loaded...')
            self._opt.load_state_dict(checkpoint['optimizer'])

            print('Previously trained optimizer state_dict loaded...')

        else:
            print("=> no checkpoint found at '{}'".format(full_name))
            quit()


    def save_checkpoint(self, save_filename):

        ''' function to record the state after each training

        Parameters
        ----------
        validation_loss : float
            validation loss per epoch
        save_filename : string
            path to the saving the checkpoint
        best_model_path : string
            path to the saving the best checkpoint
        '''

        full_name = self.check_path + save_filename
        torch.save({'model_state_dict' : self.any_network.state_dict(),
                    'optimizer': self._opt.state_dict()}, full_name)


