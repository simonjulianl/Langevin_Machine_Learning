import torch.nn as nn

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

    _reduction = 'sum' # to amplify the loss magnitude
    criterion = nn.MSELoss(reduction = _reduction)
    loss = criterion(q_quantity, q_label) + criterion(p_quantity, p_label)
    return loss