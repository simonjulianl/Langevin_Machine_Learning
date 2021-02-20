import torch
# classmethod is bound to the class and not the object of the class

class paired_distance_reduce(object):

    @classmethod
    def get_indices(self, s):

        n = s[0]
        m = torch.ones(s)
        for i in range(n):
            m[i, i, :] = 0

        return m.nonzero(as_tuple=True)

    @classmethod
    def reduce(self, src, indices):

        return src[indices]