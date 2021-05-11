import numpy as np
import torch


class paired_distance_reduce:

    def __init__(self, reduce_shape):
        self.indices = self.get_indices(reduce_shape)
        self.reduce_shape = reduce_shape

    def get_indices(self, s):

        n = s[0]
        m = torch.ones(s)
        for i in range(n):
            m[i, i, :] = 0

        return m.nonzero(as_tuple=True)

    def reduce(self, src):

        if __debug__:

            assert (src.shape == self.reduce_shape), "error in paired_distance_reduce"

        return src[self.indices]


# ============================================
if __name__ == '__main__':
    n = 3

    pair_d = paired_distance_reduce((n, n, 2))

    # now use the 'fast' method to extract the required indices
    r = torch.rand(n, n, 2)

    flatten_r = pair_d.reduce(r)
    flatten_r = flatten_r.reshape((n, n - 1, 2))  # <--- SJ add this

    print('flatten ', flatten_r)