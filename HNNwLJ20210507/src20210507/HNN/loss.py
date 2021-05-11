import torch.nn.functional as F
import torch

def qp_MSE_loss(qp_quantities, label):

    '''
    Parameters
    ----------
    predicted : tuple of length 2, with elements :
            -q_quantity : torch.tensor
            quantities related to q
            -p_quantity : torch.tensor
            quantities related to p
    label : tuple of length 2 with elements :
        -q_label : torch.tensor
        label of related q quantities
        -p_label : torch.tensor
        label of related p quantities

    Returns
    ----------
    loss : float
        Total MSE loss calculated
    '''

    q_quantity, p_quantity = qp_quantities

    q_label, p_label = label

    # print('===== predict =====')
    # print(q_quantity,p_quantity)
    # print('===== label =====')
    # print(q_label,p_label)

    if q_quantity.shape != q_label.shape or p_quantity.shape != p_label.shape:
        print('q pred, label',q_quantity.shape,q_label.shape)
        print('p pred, label',p_quantity.shape, p_label.shape)
        print('error shape not match ')
        quit()

    nsamples, nparticle, DIM = q_label.shape

    qloss = F.mse_loss(q_quantity, q_label, reduction='mean') / nparticle
    ploss = F.mse_loss(p_quantity, p_label, reduction='mean') / nparticle

    # # === for checking ===
    # q_sum = F.mse_loss(q_quantity, q_label, reduction='sum') / nparticle
    #
    # if torch.abs(q_sum / nsamples - qloss) > 1e-3 :
    #     print('mse_loss reduction error ....')
    #     quit()
    # else :
    #     print('mse_loss correct !!')
    # # === for checking ===

    loss =  qloss + ploss
    return loss,qloss,ploss

