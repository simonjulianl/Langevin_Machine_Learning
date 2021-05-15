
import torch
from torchviz import make_dot
from torch.autograd import Variable

# ======================================================
def print_compute_tree(name,node):
    dot = make_dot(node)  
    #print(dot)
    dot.render(name)
# ======================================================

if __name__=='__main__':

    torch.manual_seed(12317)
    #
    # linear y = b x + c
    #
    a  = Variable(torch.tensor([1.0]),requires_grad=True)
    b  = Variable(torch.tensor([1.0]),requires_grad=True)
    c  = Variable(torch.tensor([1.0]),requires_grad=True)
    d  = Variable(torch.tensor([1.0]),requires_grad=True)
    e  = Variable(torch.tensor([1.0]),requires_grad=True)
    f  = Variable(torch.tensor([1.0]),requires_grad=True)
    x0 = Variable(torch.tensor([.5]),requires_grad=True)

    # updater step - first step
    x1 = b*x0 + c
    x2 = x0 + x1*f # one step
    x3 = (x0-e*x1)**a + torch.sin(d*x2)
    x4 = x3*x2 + d*x1

    print_compute_tree('tree' ,x4)
