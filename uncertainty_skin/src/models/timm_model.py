from .base_model import BaseModel
import timm

class TimmModel(BaseModel):
    def build_model(self):
        if self.config.model.loss_fun == 'TM+UANLL':
            num_classes = self.config.model.num_classes + 1
        else: 
            num_classes = self.config.model.num_classes 

        model = timm.create_model(self.config.model.name, pretrained=self.config.model.pretrained, num_classes=num_classes)
        return model