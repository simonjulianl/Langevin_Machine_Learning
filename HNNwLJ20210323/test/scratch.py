class A:
    _obj_count = 0

    def __init__(self):
        A._obj_count += 1
        self.id = A._obj_count
        assert (A._obj_count == 1), type(self).__name__ + " has more than one object"


    def get_id(self):
        return self.id

import torch

if __name__=='__main__':
    # if __debug__:
    #     print('Debug ON')
    # else:
    #     print('Debug OFF')

    # a1 = A()
    # a2 = A()
    #
    # print('a1 id ',a1.get_id())
    #print('a2 id ',a2.get_id())

    ind = torch.tensor([0.9, -0.9])
    indlong =ind.type(torch.long)

    print('ind ',ind)
    print('indlong ', indlong)