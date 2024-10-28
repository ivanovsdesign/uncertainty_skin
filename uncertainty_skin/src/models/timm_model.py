from .base_model import BaseModel
import timm

class TimmModel(BaseModel):
    def build_model(self):
        model = timm.create_model(self.config.name, pretrained=self.config.pretrained, num_classes=self.config.num_classes + 1)
        return model