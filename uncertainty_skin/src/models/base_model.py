import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import distances, losses, miners, reducers, testers, regularizers

from src.functional.criterion import UANLLloss

class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self.build_model()
        self.loss_fun, self.loss_module_1 = self.build_loss()
        self.mining_func = miners.TripletMarginMiner(margin=0.2, distance=distances.CosineSimilarity(), type_of_triplets="semihard")

    def build_model(self):
        raise NotImplementedError

    def build_loss(self):
        if self.config.loss_fun == 'TM+UANLL':
            distance = distances.CosineSimilarity()
            reducer = reducers.ThresholdReducer(low=0)
            eRegularizer = regularizers.LpRegularizer()
            loss_fun = losses.TripletMarginLoss(margin=1.2, distance=distance, reducer=reducer, embedding_regularizer=eRegularizer)
            loss_module_1 = UANLLloss()
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_fun}")
        return loss_fun, loss_module_1

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.config.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'SGD':
            optimizer = optim.SGD(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        indices_tuple = self.mining_func(embeddings[:, :self.config.num_classes], y)
        loss = self.loss_fun(embeddings[:, :self.config.num_classes], y, indices_tuple) + self.loss_module_1(embeddings, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        indices_tuple = self.mining_func(embeddings[:, :self.config.num_classes], y)
        loss = self.loss_fun(embeddings[:, :self.config.num_classes], y, indices_tuple) + self.loss_module_1(embeddings, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        indices_tuple = self.mining_func(embeddings[:, :self.config.num_classes], y)
        loss = self.loss_fun(embeddings[:, :self.config.num_classes], y, indices_tuple) + self.loss_module_1(embeddings, y)
        self.log('test_loss', loss)
        return loss