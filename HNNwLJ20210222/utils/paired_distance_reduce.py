import torch
# classmethod is bound to the class and not the object of the class

class paired_distance_reduce(object):

    _obj_count = 0

    def __init__(self):

        paired_distance_reduce._obj_count += 1
        assert (paired_distance_reduce._obj_count == 1),type(self).__name__ + " has more than one object"

    @classmethod
    def get_indices(self, s):

        n = s[1]
        m = torch.ones(s)
        for i in range(n):
            m[:,i, i, :] = 0

        return m.nonzero(as_tuple=True)

    @classmethod
    def reduce(self, src, indices):

        return src[indices]