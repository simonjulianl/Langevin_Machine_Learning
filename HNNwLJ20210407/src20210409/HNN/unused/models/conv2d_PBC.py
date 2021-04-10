import torch

def compute_PBC_constants(initial_size, batch_size, initial_channels):

    def compute_constants(size, channels):
        # A, A_t are constants used to satisfy the periodic boundary condition.
        v =torch.zeros(size,)
        v[0] = 1
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
    PBC_constants['{}_{}'.format(initial_size, 1)] = compute_constants(size=initial_size, channels=1)
    PBC_constants['{}_{}'.format(initial_size, initial_channels)] = compute_constants(size=initial_size, channels=initial_channels)
    PBC_constants['{}_{}'.format(initial_size, initial_channels * 16)] = compute_constants(size=initial_size, channels=initial_channels * 16)
    PBC_constants['{}_{}'.format(initial_size, initial_channels * 32)] = compute_constants(size=initial_size, channels=initial_channels * 32)
    PBC_constants['{}_{}'.format(initial_size, initial_channels * 64)] = compute_constants(size=initial_size, channels=initial_channels * 64)
    PBC_constants['{}_{}'.format(initial_size, initial_channels * 96)] = compute_constants(size=initial_size, channels=initial_channels * 96)
    PBC_constants['{}_{}'.format(initial_size, initial_channels * 128)] = compute_constants(size=initial_size, channels=initial_channels * 128)
    PBC_constants['{}_{}'.format(initial_size // 2, initial_channels * 32)] = compute_constants(size=initial_size // 2, channels=initial_channels * 32)
    PBC_constants['{}_{}'.format(initial_size // 2, initial_channels * 64)] = compute_constants(size=initial_size // 2, channels=initial_channels * 64)
    PBC_constants['{}_{}'.format(initial_size // 2, initial_channels * 128)] = compute_constants(size=initial_size // 2, channels=initial_channels * 128)
    PBC_constants['{}_{}'.format(initial_size // 4, initial_channels * 64)] = compute_constants(size=initial_size // 4, channels=initial_channels * 64)
    PBC_constants['{}_{}'.format(initial_size // 4, initial_channels * 128)] = compute_constants(size=initial_size // 4, channels=initial_channels * 128)


    # print('PBC_constant',PBC_constants)
    return PBC_constants


def compute_PBC(inputs, pbc_constants):
    print('compute_pbc', inputs.shape)
    A_conv = pbc_constants['{}_{}'.format(inputs.shape[3], inputs.shape[1])][0]
    print(A_conv.shape)
    A_conv_t = pbc_constants['{}_{}'.format(inputs.shape[3], inputs.shape[1])][1]
    print(A_conv_t.shape)
    x = torch.matmul(torch.matmul(A_conv, inputs), A_conv_t)
    # x = torch.transpose(x, [0, 2, 3, 1])
    return x
