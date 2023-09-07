import torch
import torch.nn as nn
import torch.nn.functional as F

class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,64,10),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,128,4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128,256,4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9216, 4096),
            nn.Sigmoid()
        )
        self.output=nn.Linear(4096, 1)

    def forward(self,x1,x2):
        out1=self.conv(x1)
        out2 = self.conv(x2)
        dist=abs(out2-out1)
        out=self.output(dist)
        return out

if __name__ == '__main__':
    net=Siamese()
    print(net)
    print(list(net.parameters()))