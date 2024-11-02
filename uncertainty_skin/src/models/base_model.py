import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from pytorch_metric_learning import distances, losses, miners, reducers, testers, regularizers

from src.functional.criterion import UANLLloss

import torch

from src.utils.metrics import calculate_ece, calculate_accuracy, calculate_f1_score_binary, certain_predictions, accuracy_tta, test_vis_tta, ttac, ttaWeightedPred, hist
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

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
        
        if self.config.model.loss_fun == 'TM+UANLL':
            distance = distances.CosineSimilarity()
            reducer = reducers.ThresholdReducer(low=0)
            eRegularizer = regularizers.LpRegularizer()
            loss_fun = losses.TripletMarginLoss(margin=1.2, distance=distance, reducer=reducer, embedding_regularizer=eRegularizer)
            loss_module_1 = UANLLloss(smoothing=self.config.model.label_smoothing)
            print('UANLL loss is an additional loss term (module 1)')
        elif self.config.model.loss_fun == 'TM+CE':
            distance = distances.CosineSimilarity()
            reducer = reducers.ThresholdReducer(low=0)
            eRegularizer = regularizers.LpRegularizer()
            loss_fun = losses.TripletMarginLoss(margin=1.2, distance=distance, reducer=reducer, embedding_regularizer=eRegularizer)
            loss_module_1 = nn.CrossEntropyLoss(label_smoothing=self.config.model.label_smoothing)
            print('CE loss is an additional loss term (module 1)')
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_fun}")
        return loss_fun, loss_module_1

    def forward(self, x):
        return self.model(x)
    
    def lr_lambda(self,epoch,n=150,
              delay=30,stop_lr=0):
        n = self.config.trainer.max_epochs
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
        indices_tuple = self.mining_func(embeddings[:, :self.config.model.num_classes], y)
        loss = self.loss_fun(embeddings[:, :self.config.model.num_classes], y, indices_tuple) + self.loss_module_1(embeddings, y)
        if self.config.loss_fun == 'TM+CE':
            preds = nn.functional.softmax(embeddings[:, :self.config.model.num_classes], 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        else:
            acc = (embeddings[:, :self.config.model.num_classes].argmax(dim=-1) == y).float().mean()
        self.log("train_loss", loss.float(), on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        indices_tuple = self.mining_func(embeddings[:, :self.config.model.num_classes], y)
        loss = self.loss_fun(embeddings[:, :self.config.model.num_classes], y, indices_tuple) + self.loss_module_1(embeddings, y)
        if self.config.loss_fun == 'TM+CE':
            preds = nn.functional.softmax(embeddings[:, :self.config.model.num_classes], 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        else:
            acc = (embeddings[:, :self.config.model.num_classes].argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss.float(), on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        if self.config.model.loss_fun == 'TM+CE':
            preds = nn.functional.softmax(embeddings[:, :self.config.model.num_classes], 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        else:
            acc = (embeddings[:, :self.config.model.num_classes].argmax(dim=-1) == y).float().mean()
        print("Test set accuracy (Precision@1) = {}".format(acc))
        self.log("test_acc", acc, on_epoch=True)

    def on_test_epoch_end(self):
        # Perform TTA
        num_tta = self.config.dataset.num_tta
        test_predictions_tta = []
        test_labels_tta = []
        test_certainties_s_tta = []
        test_confidences_s_tta = []
        for _ in range(num_tta):
            test_predictions, test_labels, test_certainties_s, test_confidences_s = [], [], [], []
            for batch in self.trainer.datamodule.test_dataloader():
                inputs, labels = batch
                inputs = inputs.to(self.device)  # Move inputs to the same device as the model
                outputs = self(inputs)
                preds = torch.softmax(outputs[:, :self.config.model.num_classes], dim=1)
                certainties_s = torch.exp(-outputs[:, self.config.model.num_classes:])
                confidences_s = preds.max(dim=1).values
                test_predictions.append(preds)
                test_labels.append(labels)
                test_certainties_s.append(certainties_s)
                test_confidences_s.append(confidences_s)

            test_predictions = torch.cat(test_predictions)
            test_labels = torch.cat(test_labels)
            test_certainties_s = torch.cat(test_certainties_s)
            test_confidences_s = torch.cat(test_confidences_s)

            test_predictions_tta.append(test_predictions)
            test_labels_tta.append(test_labels)
            test_certainties_s_tta.append(test_certainties_s)
            test_confidences_s_tta.append(test_confidences_s)

        # Collect and log metrics
        test_predictions_tta = torch.stack(test_predictions_tta).mean(dim=0)
        test_labels_tta = torch.stack(test_labels_tta).mode(dim=0).values
        test_certainties_s_tta = torch.stack(test_certainties_s_tta).mean(dim=0)
        test_confidences_s_tta = torch.stack(test_confidences_s_tta).mean(dim=0)

        test_predictions_tta = test_predictions_tta.argmax(dim=1).cpu()  # Move to CPU
        test_labels_tta = test_labels_tta.squeeze().cpu()  # Move to CPU

        accuracy = accuracy_score(test_labels_tta, test_predictions_tta)
        f1 = f1_score(test_labels_tta, test_predictions_tta, average='weighted')
        precision = precision_score(test_labels_tta, test_predictions_tta, average='weighted')
        recall = recall_score(test_labels_tta, test_predictions_tta, average='weighted')
        roc_auc = roc_auc_score(test_labels_tta, test_predictions_tta, average='weighted')

        self.logger.log_metrics({
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        })

        # Save predictions
        predictions_df = pd.DataFrame({
            'true_labels': test_labels_tta.tolist(),
            'predictions': test_predictions_tta.tolist(),
            'certainties_s': test_certainties_s_tta.cpu().tolist(),  # Move to CPU
            'confidences_s': test_confidences_s_tta.cpu().tolist()  # Move to CPU
        })
        predictions_df.to_csv(f"{self.config.model.name}_predictions.csv", index=False)

        # Plot confusion matrix
        cm = confusion_matrix(test_labels_tta, test_predictions_tta)
        cm_display = ConfusionMatrixDisplay(cm, display_labels=self.config.dataset.class_names).plot()
        plt.savefig(f"{self.config.model.name}_confusion_matrix.png")

        # Calculate TTA metrics
        test_attr_tta = test_vis_tta(self,
                                     self.trainer.datamodule.test_dataloader(),
                                     num_classes = self.config.model.num_classes,
                                     loss_fun = self.config.model.loss_fun,
                                     figs=5,
                                     numTTA=self.config.dataset.num_tta)
        
        mode_labels = torch.mode(test_attr_tta['labels_tta'], dim=0).values
        mode_predictions = torch.mode(test_attr_tta['predictions_tta'], dim=0).values
        mode_confidences_s_tta = torch.mode(test_attr_tta['confidences_s_tta'], dim=0).values
        mode_certainties_s_tta = torch.mode(test_attr_tta['certainties_s_tta'], dim=0).values

        true_pred = (mode_labels == mode_predictions).tolist()
        df = pd.DataFrame({
            'Mode_label_tta': mode_labels,
            'Mode_prediction_tta': mode_predictions,
            'Mode_confidence_(soft)': mode_confidences_s_tta,
            'Mode_certainty_(soft)': mode_certainties_s_tta,
            'True or false prediction': true_pred
        })

        # Plot histograms
        hist(df, 'Mode_confidence_(soft)', (4, 4))
        hist(df, 'Mode_certainty_(soft)', (4, 4))

        # Confusion matrix for mode-based TTA predictions
        cm = confusion_matrix(mode_labels.cpu(), mode_predictions.cpu())
        print('Mode based TTA predictions (TTAM)')
        cm_display = ConfusionMatrixDisplay(cm, display_labels=self.config.dataset.class_names).plot()

        # Weighted TTA predictions
        weightedPred = ttaWeightedPred(test_attr_tta['labels_tta'], test_attr_tta['predictions_tta'], test_attr_tta['confidences_s_tta'], test_attr_tta['certainties_s_tta'], num_classes=self.config.model.num_classes)
        n_correct_TTAWCo_S = (weightedPred['predictionsCo'] == mode_labels).sum().item()
        n_correct_TTAWCe_S = (weightedPred['predictionsCe'] == mode_labels).sum().item()

        n_samples = mode_labels.shape[0]
        accuracy_TTAWCo_S = n_correct_TTAWCo_S / n_samples
        accuracy_TTAWCe_S = n_correct_TTAWCe_S / n_samples

        print('---------------------- Soft TTA predictions -----------------------------')
        print('n_samples', n_samples)
        print(f'accuracy of confidences based soft TTA predictions = {accuracy_TTAWCo_S}')
        print(f'accuracy of certainties based soft TTA predictions = {accuracy_TTAWCe_S}')

        # Confusion matrix for confidence-based soft TTA predictions
        cm = confusion_matrix(mode_labels.cpu(), weightedPred['predictionsCo'].cpu())
        print('Confidence based soft TTA predictions (TTAWCo-S)')
        cm_display = ConfusionMatrixDisplay(cm, display_labels=self.config.dataset.class_names).plot()

        # Summary table
        test_accuracy_summary = {
            'ID': self.logger._task.id,
            'seed': self.config.dataset.seed,
            '# samples': len(test_labels_tta),
            '# TTA': self.config.dataset.num_tta,
            'Seed': self.config.dataset.seed,
            'F1 (without TTA)': f1,
            'Acc (without TTA)': accuracy,
            'Acc TTAM': accuracy_tta(mode_labels, mode_predictions)['acc_without_u'],
            'Acc TTAWCo-S': accuracy_TTAWCo_S,
            'Acc TTAWCe-S': accuracy_TTAWCe_S,
            'ECE': calculate_ece(test_predictions_tta, test_labels_tta, num_classes=self.config.model.num_classes)
        }

        summary_df = pd.DataFrame([test_accuracy_summary])
        summary_df.to_csv(f"{self.config.model.name}_summary.csv", index=False)

        # Calculate std/error for metrics across seeds
        # Assuming you have a list of summary dataframes from different seeds
        # summary_dfs = [summary_df1, summary_df2, ...]
        # combined_summary_df = pd.concat(summary_dfs)
        # metrics_std = combined_summary_df.std()
        # metrics_error = combined_summary_df.sem()
