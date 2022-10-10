import torch
import torch.nn as nn
import torchvision

class Model(nn.Module):
    def __init__(self, Nh, Nw, bs, ptsz = 32, pout = 512):
        super(Model, self).__init__()

        self.Nh = Nh
        self.Nw = Nw
        self.bs = bs
        self.ptsz = ptsz
        self.pout = pout
        self.base_encoder = torchvision.models.resnet18(pretrained=False)
        self.base_encoder.fc = nn.Identity()

        self.proj2 = nn.Sequential(*[nn.Linear(512, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, self.pout),
                                    nn.BatchNorm1d(self.pout)])
        
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.cntH = self.Nh//(self.ptsz//2) - 1
        self.cntW = self.Nw//(self.ptsz//2) - 1
    
    def forward(self, x):
        x = x.view((-1,3,self.ptsz,self.ptsz))
        x = self.base_encoder(x)
        #print(x.shape)
        x = x.view((self.bs, -1, self.cntH, self.cntW))
        #print(x.shape)
        #x = self.proj1(x)
        #print(x.shape)
        x = self.gap(x).squeeze()
        #print(x.shape)
        x1 = self.proj2(x)
        
        return x, x1