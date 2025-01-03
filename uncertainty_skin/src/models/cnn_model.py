from .base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class CNN(BaseModel):
    def build_model(self):
        input_channel = self.config.model.input_channel
        if self.config.model.loss_fun in ['TM+UANLL', 'UANLL']:
            print('CNN. Uncertainty toggled')
            n_outputs = self.config.model.num_classes + 1
        else:
            n_outputs = self.config.model.num_classes

        dropout_rate = self.config.model.dropout_rate
        top_bn = self.config.model.top_bn

        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, n_outputs)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)
        
        # Define batch normalization for the linear layer if needed
        if self.top_bn:
            self.bn_c1 = nn.BatchNorm1d(n_outputs)

    def call_bn(self, bn, x):
        return bn(x)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(self.call_bn(self.bn1, h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(self.call_bn(self.bn2, h), negative_slope=0.01)
        h = self.c3(h)
        h = F.leaky_relu(self.call_bn(self.bn3, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(self.call_bn(self.bn4, h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(self.call_bn(self.bn5, h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(self.call_bn(self.bn6, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(self.call_bn(self.bn7, h), negative_slope=0.01)
        h = self.c8(h)
        h = F.leaky_relu(self.call_bn(self.bn8, h), negative_slope=0.01)
        h = self.c9(h)
        h = F.leaky_relu(self.call_bn(self.bn9, h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit = self.l_c1(h)
        if self.top_bn:
            logit = self.call_bn(self.bn_c1, logit)
        return logit

    def intermediate_forward(self, x):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(self.call_bn(self.bn1, h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(self.call_bn(self.bn2, h), negative_slope=0.01)
        h = self.c3(h)
        h = F.leaky_relu(self.call_bn(self.bn3, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(self.call_bn(self.bn4, h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(self.call_bn(self.bn5, h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(self.call_bn(self.bn6, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        h = F.leaky_relu(self.call_bn(self.bn7, h), negative_slope=0.01)
        h = self.c8(h)
        h = F.leaky_relu(self.call_bn(self.bn8, h), negative_slope=0.01)
        h = self.c9(h)
        h = F.leaky_relu(self.call_bn(self.bn9, h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        return h