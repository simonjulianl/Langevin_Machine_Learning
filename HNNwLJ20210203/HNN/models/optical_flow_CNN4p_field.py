import torch
import torch.nn as nn

class optical_flow_CNN4p_field(nn.Module):

    def __init__(self):

        super(optical_flow_CNN4p_field, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


    def forward(self,data): # data -> del_list ( del_qx, del_qy, del_px, del_py, t )

        out = self.layer1(data)
        out = self.layer2(out)

        return out