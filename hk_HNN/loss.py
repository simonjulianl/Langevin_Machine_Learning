import torch.nn as nn

def qp_MSE_loss(qp_quantities, label):

    q_quantity, p_quantity = qp_quantities

    q_label, p_label = label

    print('=== loss ===')
    print('=== predict ===')
    print(q_quantity,p_quantity)
    print('=== label ===')
    print(q_label,p_label)
    print(q_quantity.shape,p_quantity.shape)
    print(q_label.shape,p_label.shape)

    _reduction = 'sum' # to amplify the loss magnitude
    criterion = nn.MSELoss(reduction = _reduction)
    loss = criterion(q_quantity, q_label) + criterion(p_quantity, p_label)
    return loss