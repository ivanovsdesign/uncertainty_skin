from .base_model import BaseModel
import timm

class TimmModel(BaseModel):
    def build_model(self):
        if self.config.loss_fun == 'TM+UANLL':
            num_classes = self.config.num_classes + 1
        else: 
            num_classes = self.config.num_classes 

        model = timm.create_model(self.config.name, pretrained=self.config.pretrained, num_classes=num_classes)
        return model