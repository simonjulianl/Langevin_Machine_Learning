import torch
from MD_parameters import MD_parameters

class dataset:

    ''' dataset class to help split data '''

    _obj_count = 0

    def __init__(self):

        dataset._obj_count += 1
        assert(dataset._obj_count == 1), type(self).__name__ + ' has more than one object'


    def _shuffle(self, q_list, p_list):
        # for internal use only

        idx = torch.randperm(q_list.shape[0])

        q_list_shuffle = q_list[idx]
        p_list_shuffle = p_list[idx]

        try:
            assert q_list_shuffle.shape == p_list_shuffle.shape
        except:
             raise Exception('does not have shape method or shape differs')

        return q_list_shuffle, p_list_shuffle


    def qp_dataset(self, qp_list, qp_crash = None, shuffle = True):

        ''' given mode, split data for train (shuffle=True) or test (shuffle=False)

        Parameters
        ----------
        shuffle : bool, optional
                True when train and valid / False when test

        Returns
        ----------
        inital q and p
        shape is [ (q,p), nsamples, nparticle, DIM ]
        '''

        q_list1 = qp_list[0]; p_list1 = qp_list[1]

        if shuffle:
            q_list1, p_list1 = self._shuffle(q_list1, p_list1)

        if qp_crash is not None:

            q_crash = qp_crash[0]; p_crash = qp_crash[1]

            print('n. of crash data', q_list1.shape, 'n. of original train data', q_list1.shape)

            y = int(MD_parameters.crash_duplicate_ratio * len(q_list1) / len(q_crash)) # duplicate crash data
            z = len(q_list1) - y * len(q_crash)  # reduced train data

            print('crash duplicate', y, 'reduced train data', z)

            indices = torch.randperm(len(q_list1))[:z]

            q_reduced = q_list1[indices]
            p_reduced = p_list1[indices]

            q_duplicate_crash = q_crash.repeat(y,1,1)
            p_duplicate_crash = p_crash.repeat(y,1,1)
            # print(q_duplicate_crash , p_duplicate_crash )

            q_list2 = torch.cat((q_reduced, q_duplicate_crash.cpu()), dim=0)
            p_list2 = torch.cat((p_reduced, p_duplicate_crash.cpu()), dim=0)

            q_list_shuffle, p_list_shuffle = self._shuffle(q_list2, p_list2)

        else:

            q_list_shuffle = q_list1
            p_list_shuffle = p_list1

        qp_list_shuffle = torch.stack((q_list_shuffle,p_list_shuffle))

        return qp_list_shuffle
