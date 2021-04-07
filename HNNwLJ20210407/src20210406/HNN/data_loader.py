import torch
from utils.data_io import data_io
from torch.utils.data import Dataset

# qp_list.shape = [nsamples, (q,p)=2, trajectory=2 (input,label), nparticle, DIM]
# ===========================================================
class torch_dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, filename):
        """
        Args:
            filename (string): Numpy file for data and label
        """
        qp_list, tau_short, tau_long, boxsize = data_io.read_trajectory_qp(filename)
        # qp_list.shape = [nsamples, (q,p), trajectory (input,label), nparticle, DIM]

        self.qp_list_input   = qp_list[:,:,0,:,:]
        self.qp_list_label   = qp_list[:,:,1,:,:]
        self.data_tau_short = tau_short
        self.data_tau_long  = tau_long
        self.data_boxsize   = boxsize

    def __len__(self):
        ''' Denotes the total number of samples '''
        return self.qp_list_input.shape[0]

    def __getitem__(self, idx):
        ''' Generates one sample of data '''
        # Select sample
        if idx >= self.__len__():
            raise ValueError('idx ' + str(idx) +' exceed length of data: ' + str(self.__len__()))

        return self.qp_list_input[idx], self.qp_list_label[idx]

# ===========================================================
class my_data:
    def __init__(self,train_filename, val_filename, test_filename, train_pts=0, val_pts=0, test_pts=0):

        self.train_set = torch_dataset(train_filename)
        self.val_set   = torch_dataset(val_filename)
        self.test_set  = torch_dataset(test_filename)

        # perform subsampling of data when specified
        # this is important when we need to perform quick debugging with
        # few data points
        if train_pts > 0:
            if train_pts > len(self.train_set):
                print('available ', len(self.train_set))
                raise ValueError("ERROR: request more than subspace set")
            self.train_set = self.sample(self.train_set, train_pts)

        if val_pts > 0:
            if val_pts > len(self.val_set):
                print('available ', len(self.val_set))
                raise ValueError("ERROR: request more than subspace set")
            self.val_set = self.sample(self.val_set, val_pts)

        if test_pts > 0:
            if test_pts > len(self.test_set):
                print('available ', len(self.test_set))
                raise ValueError("ERROR: request more than subspace set")
            self.test_set = self.sample(self.test_set, test_pts)

    # ===========================================================
    # sample data_set with num_pts of points
    # ===========================================================
    def sample(self, data_set, num_pts):

        # perform subsampling of data when specified
        # this is important when we need to perform quick debugging with
        # few data points
        if num_pts > 0:
            if num_pts > len(data_set):
                print("error: request more than CIFAR10 set")
                print('available ',len(self.train_set))
                quit()
 
        data_set.qp_list_input = data_set.qp_list_input[:num_pts]
        data_set.qp_list_label = data_set.qp_list_label[:num_pts] 

        return data_set

# ===========================================================
#
class data_loader:
    #  data loader upon this custom dataset
    def __init__(self,data_set, batch_size):

        self.data_set = data_set

        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 2, 'pin_memory': False} if use_cuda else {}
        # num_workers: the number of processes that generate batches in parallel.
        print('kwargs ',kwargs)

        self.train_loader = torch.utils.data.DataLoader(self.data_set.train_set,
                            batch_size=batch_size, shuffle=True, **kwargs)

        self.val_loader   = torch.utils.data.DataLoader(self.data_set.val_set,
                            batch_size=batch_size, shuffle=True, **kwargs)

        self.test_loader  = torch.utils.data.DataLoader(self.data_set.test_set,
                            batch_size=batch_size, shuffle=True, **kwargs)


