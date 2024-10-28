from .base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class CNN(BaseModel):
    def build_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, self.config.num_classes + 1)
        )