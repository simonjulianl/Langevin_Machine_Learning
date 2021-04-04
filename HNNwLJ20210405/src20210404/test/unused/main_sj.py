import torch


def paired_distance_reduced(q, nparticle, DIM):

    print("==pb==")
    qlen = q.shape[0]
    q0 = torch.unsqueeze(q, dim=0)
    qm = torch.repeat_interleave(q0, qlen, dim=0)
    qt = qm.permute(1, 0, 2)

    dq = qt - qm
    print('dq', dq)
    indices = torch.where(torch.abs(dq) > 0.5)
    dq[indices] = dq[indices] - torch.round(dq[indices])

    dq = dq[dq.nonzero(as_tuple=True)].reshape(nparticle, nparticle - 1, DIM)

    return dq

if __name__ == '__main__':

    sample = 1
    nparticle = 4
    DIM = 2
    q = torch.rand(nparticle, DIM)

    dq = paired_distance_reduced(q, nparticle, DIM)
    print('flatten ', dq)