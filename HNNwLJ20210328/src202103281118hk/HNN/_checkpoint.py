import os
import torch
import shutil

class _checkpoint:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, any_network, opt, current_epoch):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.any_network = any_network
        self._opt = opt
        self._current_epoch = current_epoch

    def load_checkpoint(self, load_path):

        ''' function to load saved model

        Parameters
        ----------
        load_path : string
                load saved model. if not, pass
        '''

        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            checkpoint = torch.load(load_path)[0]
            # print(checkpoint)
            # load models weights state_dict
            self.any_network.load_state_dict(checkpoint['model_state_dict'])
            print('Previously trained models weights state_dict loaded...')
            self._opt.load_state_dict(checkpoint['optimizer'])
            print('Previously trained optimizer state_dict loaded...')
            # self._scheduler = checkpoint['scheduler']
            # print('Previously trained scheduler state_dict loaded...')
            self._current_epoch = checkpoint['epoch'] + 1
            print('Previously trained epoch state_dict loaded...')
            print('current_epoch', self._current_epoch)

            if not os.path.exists('./retrain_saved_model/'):
                os.makedirs('./retrain_saved_model/')

        else:
            print("=> no checkpoint found at '{}'".format(load_path))

            if not os.path.exists('./saved_model/'):
                os.makedirs('./saved_model/')
            # # epoch, best_precision, loss_train
            # return 1, 0, []

    def save_checkpoint(self, validation_loss, save_path, best_model_path, current_epoch):

        ''' function to record the state after each training

        Parameters
        ----------
        validation_loss : float
            validation loss per epoch
        save_path : string
            path to the saving the checkpoint
        best_model_path : string
            path to the saving the best checkpoint
        '''

        # initialize best models
        self._best_validation_loss = float('inf')
        self._current_epoch = current_epoch
        is_best = validation_loss < self._best_validation_loss
        self._best_validation_loss = min(validation_loss, self._best_validation_loss)

        torch.save(({
                'epoch': self._current_epoch,
                'model_state_dict' : self.any_network.state_dict(),
                'best_validation_loss' : self._best_validation_loss,
                'optimizer': self._opt.state_dict(),
                # 'scheduler' : self._scheduler
                }, is_best), save_path)

        if is_best:
            shutil.copyfile(save_path, best_model_path)
