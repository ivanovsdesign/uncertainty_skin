from .base_model import BaseModel
import timm

class TimmModel(BaseModel):
    def build_model(self):
        if self.config.model.loss_fun in ['TM+UANLL', 'UANLL']:
            num_classes = self.config.model.num_classes + 1
        else:
            num_classes = self.config.model.num_classes

        self.model = timm.create_model(self.config.model.name, pretrained=self.config.model.pretrained, num_classes=num_classes)
        
        return self.model

    def forward(self, x):
        return self.model(x)

    def intermediate_forward(self, x):
        # Assuming the last layer before the classifier is the penultimate layer
        for name, module in self.model.named_children():
            if name == 'head':
                break
            x = module(x)
        return x