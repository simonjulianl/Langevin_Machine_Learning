import torch

def compute_PBC_constants(initial_size, batch_size, in_channels):
    '''
    this function to do pbc padding

    :param initial_size: initial input size
    :param batch_size: batch size
    :param in_channels:
    '''

    def compute_constants(size, channels):
        # A, A_t are constants used to satisfy the periodic boundary condition.
        v =torch.zeros(size,)
        v[0] = 1

        # Reverse the order of a tensor along given axis in dims.
        v_flip = torch.flip(v, dims=(0,))

        v = torch.unsqueeze(v, dim=0)
        v_flip = torch.unsqueeze(v_flip, dim=0)

        b = torch.cat([v_flip, torch.eye(size), v])
        b_t = torch.transpose(b, 0, 1)

        b = b.unsqueeze(0).repeat(batch_size, 1, 1)
        b_t = b_t.unsqueeze(0).repeat(batch_size, 1, 1)
        A_conv = b.unsqueeze(1).repeat(1, channels, 1, 1)
        A_conv_t = b_t.unsqueeze(1).repeat(1, channels, 1, 1)

        return [A_conv, A_conv_t]

    PBC_constants = {}
    PBC_constants['{}_{}'.format(initial_size, 2)] = compute_constants(size=initial_size, channels=2)
    PBC_constants['{}_{}'.format(initial_size, in_channels)] = compute_constants(size=initial_size, channels=in_channels)
    PBC_constants['{}_{}'.format(initial_size // 2, in_channels)] = compute_constants(size=initial_size// 2, channels=in_channels)
    PBC_constants['{}_{}'.format(initial_size // 4, in_channels)] = compute_constants(size=initial_size // 4, channels=in_channels)
    PBC_constants['{}_{}'.format(initial_size // 8, in_channels)] = compute_constants(size=initial_size // 8, channels=in_channels)
    PBC_constants['{}_{}'.format(initial_size // 16, in_channels)] = compute_constants(size=initial_size // 16, channels=in_channels)
    PBC_constants['{}_{}'.format(initial_size // 2, in_channels * 2)] = compute_constants(size=initial_size // 2, channels=in_channels * 2)
    PBC_constants['{}_{}'.format(initial_size // 4, in_channels * 2)] = compute_constants(size=initial_size // 4, channels=in_channels * 2)
    PBC_constants['{}_{}'.format(initial_size // 8, in_channels * 2)] = compute_constants(size=initial_size // 8, channels=in_channels * 2)
    PBC_constants['{}_{}'.format(initial_size // 16, in_channels * 2)] = compute_constants(size=initial_size // 16, channels=in_channels * 2)
    # print('PBC_constant',PBC_constants)
    return PBC_constants


def compute_PBC(inputs, pbc_constants):

    A_conv = pbc_constants['{}_{}'.format(inputs.shape[3], inputs.shape[1])][0]
    # shape is [batch_size, channels, gridL+2, gridL]

    A_conv_t = pbc_constants['{}_{}'.format(inputs.shape[3], inputs.shape[1])][1]
    # shape is [batch_size, channels, gridL, gridL+2]

    x = torch.matmul(torch.matmul(A_conv, inputs), A_conv_t)
    # shape is [batch_size, channels, gridL+2, gridL+2] ; padding = 2

    return x
