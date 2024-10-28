import torch
import torch.nn as nn

class UANLLloss(nn.Module):
    def __init__(self, smoothing = 0.1, classes=2):
        super(UANLLloss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.clc = classes
        
    def forward(self,x,y):
        logvar  = (x[:,self.clc:]) #** 2 
        prob = x[:,:self.clc]

        with torch.no_grad():
            yoh = torch.zeros_like(prob)
            yoh.fill_(self.smoothing / (self.clc - 1))
            yoh.scatter_(1, y.data.unsqueeze(1), self.confidence)

        loss0 = ((yoh - prob) ** 2).sum(dim=1)
        loss = (torch.exp(-logvar) * loss0 + self.clc * logvar)

        return loss.mean()