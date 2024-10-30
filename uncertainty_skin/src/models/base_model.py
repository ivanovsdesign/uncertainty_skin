import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import distances, losses, miners, reducers, testers, regularizers

from src.functional.criterion import UANLLloss

import torch

class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = self.build_model()
        self.loss_fun, self.loss_module_1 = self.build_loss()
        self.mining_func = miners.TripletMarginMiner(margin=0.2, distance=distances.CosineSimilarity(), type_of_triplets="semihard")

        self.epoch_val_loss = []
        self.epoch_train_loss = []

    def build_model(self):
        raise NotImplementedError

    def build_loss(self):
        
        if self.config.loss_fun == 'TM+UANLL':
            distance = distances.CosineSimilarity()
            reducer = reducers.ThresholdReducer(low=0)
            eRegularizer = regularizers.LpRegularizer()
            loss_fun = losses.TripletMarginLoss(margin=1.2, distance=distance, reducer=reducer, embedding_regularizer=eRegularizer)
            loss_module_1 = UANLLloss(smoothing=self.config.label_smoothing)
            print('UANLL loss is an additional loss term (module 1)')
        elif self.config.loss_fun == 'TM+CE':
            distance = distances.CosineSimilarity()
            reducer = reducers.ThresholdReducer(low=0)
            eRegularizer = regularizers.LpRegularizer()
            loss_fun = losses.TripletMarginLoss(margin=1.2, distance=distance, reducer=reducer, embedding_regularizer=eRegularizer)
            loss_module_1 = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
            print('CE loss is an additional loss term (module 1)')
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_fun}")
        return loss_fun, loss_module_1

    def forward(self, x):
        return self.model(x)
    
    def lr_lambda(self,epoch,n=150,
              delay=30,stop_lr=0):
        start_lr = self.config.lr
        learning_rate = start_lr if epoch < delay else start_lr - (epoch - delay) * (start_lr - stop_lr) / (n - 1 - delay)
        return learning_rate / start_lr

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.config.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.config.optimizer_hparams)
        elif self.config.optimizer == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.config.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.config.optimizer_name}"'

        # We will reduce the learning rate by 'gamma' after 'milestone' epochs
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)#MultiStepLR(optimizer, milestones=MILESTONES, gamma=GAMMA)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        indices_tuple = self.mining_func(embeddings[:, :self.config.num_classes], y)
        loss = self.loss_fun(embeddings[:, :self.config.num_classes], y, indices_tuple) + self.loss_module_1(embeddings, y)
        if self.config.loss_fun == 'TM+CE':
            preds = nn.functional.softmax(embeddings[:, :self.config.num_classes], 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        else:
            acc = (embeddings[:, :self.config.num_classes].argmax(dim=-1) == y).float().mean()
        self.log("train_loss", loss.float(), on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        indices_tuple = self.mining_func(embeddings[:, :self.config.num_classes], y)
        loss = self.loss_fun(embeddings[:, :self.config.num_classes], y, indices_tuple) + self.loss_module_1(embeddings, y)
        if self.config.loss_fun == 'TM+CE':
            preds = nn.functional.softmax(embeddings[:, :self.config.num_classes], 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        else:
            acc = (embeddings[:, :self.config.num_classes].argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss.float(), on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        if self.config.loss_fun == 'TM+CE':
            preds = nn.functional.softmax(embeddings[:, :self.config.num_classes], 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        else:
            acc = (embeddings[:, :self.config.num_classes].argmax(dim=-1) == y).float().mean()
        print("Test set accuracy (Precision@1) = {}".format(acc))
        self.log("test_acc", acc, on_epoch=True)