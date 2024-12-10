from pytorch_metric_learning import distances, losses, miners, reducers, testers, regularizers

from src.functional.criterion import UANLLloss

from src.utils.metrics import calculate_ece, calculate_accuracy, calculate_f1_score_binary, certain_predictions, accuracy_tta, test_vis_tta, ttac, ttaWeightedPred, hist

from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.build_model()
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
            loss_fun = losses.TripletMarginLoss(margin=self.config.model.margin, distance=distance, reducer=reducer, embedding_regularizer=eRegularizer)
            loss_module_1 = UANLLloss(smoothing=self.config.model.label_smoothing)
            print('UANLL loss is an additional loss term (module 1)')
        elif self.config.model.loss_fun == 'TM+CE':
            distance = distances.CosineSimilarity()
            reducer = reducers.ThresholdReducer(low=0)
            eRegularizer = regularizers.LpRegularizer()
            loss_fun = losses.TripletMarginLoss(margin=self.config.model.margin, distance=distance, reducer=reducer, embedding_regularizer=eRegularizer)
            loss_module_1 = nn.CrossEntropyLoss(label_smoothing=self.config.model.label_smoothing)
            print('CE loss is an additional loss term (module 1)')
        elif self.config.model.loss_fun == 'CE':
            loss_fun = None
            loss_module_1 = nn.CrossEntropyLoss(label_smoothing=self.config.model.label_smoothing)
            print('Single CE loss')
        elif self.config.model.loss_fun == 'UANLL':
            loss_fun = None
            loss_module_1 = UANLLloss(smoothing=self.config.model.label_smoothing)
            print('Single UANLL loss')
        else:
            raise ValueError(f"Unknown loss function: {self.config.model.loss_fun}")
        return loss_fun, loss_module_1

    def forward(self, x):
        return self.model(x)
    
    def lr_lambda(self, epoch, n=150, delay=30, stop_lr=0):
        n = self.config.trainer.max_epochs
        start_lr = self.config.model.lr
        learning_rate = start_lr if epoch < delay else start_lr - (epoch - delay) * (start_lr - stop_lr) / (n - 1 - delay)
        return learning_rate / start_lr

    def configure_optimizers(self):
        if self.config.model.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.config.model.optimizer_hparams)
        elif self.config.model.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.config.model.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.config.model.optimizer_name}"'

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        if self.loss_fun is not None:
            indices_tuple = self.mining_func(embeddings[:, :self.config.model.num_classes], y)
            loss = self.loss_fun(embeddings[:, :self.config.model.num_classes], y, indices_tuple)
        else:
            loss = 0
        loss += self.loss_module_1(embeddings, y)
        if self.config.model.loss_fun in ['TM+CE', 'CE']:
            preds = nn.functional.softmax(embeddings, 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        elif self.config.model.loss_fun in ['TM+UANLL', 'UANLL']:
            preds = nn.functional.softmax(embeddings[:, :self.config.model.num_classes], 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        self.log("train_loss", loss.float(), on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        if self.loss_fun is not None:
            indices_tuple = self.mining_func(embeddings[:, :self.config.model.num_classes], y)
            loss = self.loss_fun(embeddings[:, :self.config.model.num_classes], y, indices_tuple)
        else:
            loss = 0
        loss += self.loss_module_1(embeddings, y)
        if self.config.model.loss_fun in ['TM+CE', 'CE']:
            preds = nn.functional.softmax(embeddings[:, :self.config.model.num_classes], 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        else:
            acc = (embeddings[:, :self.config.model.num_classes].argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss.float(), on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        embeddings = self(x)
        if self.config.model.loss_fun in ['TM+CE', 'CE']:
            preds = nn.functional.softmax(embeddings[:, :self.config.model.num_classes], 1)
            acc = (preds.argmax(dim=-1) == y).float().mean()
        else:
            acc = (embeddings[:, :self.config.model.num_classes].argmax(dim=-1) == y).float().mean()
        print("Test set accuracy without TTA (Precision@1) = {}".format(acc))
        self.log("test_acc", acc, on_epoch=True)

    def on_test_epoch_end(self):
        num_tta = self.config.dataset.num_tta
        test_predictions_tta = []
        test_labels_tta = []
        test_certainties_s_tta = []
        test_confidences_s_tta = []

        # Collect predictions without TTA
        test_predictions_no_tta, test_labels_no_tta = [], []
        for batch in self.trainer.datamodule.test_dataloader():
            inputs, labels = batch
            inputs = inputs.to(self.device)
            outputs = self(inputs)
            preds = torch.softmax(outputs[:, :self.config.model.num_classes], dim=1)
            test_predictions_no_tta.append(preds)
            test_labels_no_tta.append(labels)

        test_predictions_no_tta = torch.cat(test_predictions_no_tta)
        test_labels_no_tta = torch.cat(test_labels_no_tta)
        test_probs_no_tta = test_predictions_no_tta.squeeze().cpu()
        test_predictions_no_tta = test_predictions_no_tta.argmax(dim=1).cpu()
        test_labels_no_tta = test_labels_no_tta.squeeze().cpu()
        
        print(f'Test labels no TTA shape: {test_labels_no_tta.shape}')
        print(f'Test probs no TTA shape: {test_probs_no_tta.argmax(-1).shape}')

        accuracy_no_tta = accuracy_score(test_labels_no_tta, test_predictions_no_tta)
        f1_no_tta = f1_score(test_labels_no_tta, test_predictions_no_tta, average='weighted')
        precision_no_tta = precision_score(test_labels_no_tta, test_predictions_no_tta, average='weighted')
        recall_no_tta = recall_score(test_labels_no_tta, test_predictions_no_tta, average='weighted')
        roc_auc_no_tta = roc_auc_score(test_labels_no_tta, test_probs_no_tta.argmax(-1), average='weighted', multi_class='ovr')

        print(f'Metrics without TTA: Accuracy={accuracy_no_tta}, F1={f1_no_tta}, Precision={precision_no_tta}, Recall={recall_no_tta}, ROC-AUC={roc_auc_no_tta}')

        for _ in tqdm(range(num_tta), total=num_tta):
            test_predictions, test_labels, test_certainties_s, test_confidences_s = [], [], [], []
            for batch in self.trainer.datamodule.tta_dataloader():
                inputs, labels = batch
                inputs = inputs.to(self.device)
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

        print('Collecting and logging metrics')

        test_predictions_tta = torch.stack(test_predictions_tta).mean(dim=0)
        test_labels_tta = torch.stack(test_labels_tta).mode(dim=0).values
        test_certainties_s_tta = torch.stack(test_certainties_s_tta).mean(dim=0)
        test_confidences_s_tta = torch.stack(test_confidences_s_tta).mean(dim=0)
        test_probs_tta = test_predictions_tta.squeeze().cpu()
        test_predictions_tta = test_predictions_tta.argmax(dim=1).cpu()
        test_labels_tta = test_labels_tta.squeeze().cpu()

        accuracy = accuracy_score(test_labels_tta, test_predictions_tta)
        f1 = f1_score(test_labels_tta, test_predictions_tta, average='weighted')
        precision = precision_score(test_labels_tta, test_predictions_tta, average='weighted')
        recall = recall_score(test_labels_tta, test_predictions_tta, average='weighted')
        roc_auc = roc_auc_score(test_labels_tta, test_probs_tta.argmax(-1), average='weighted', multi_class='ovr')

        print(f'Metrics with TTA: Accuracy={accuracy}, F1={f1}, Precision={precision}, Recall={recall}, ROC-AUC={roc_auc}')

        self.logger.log_metrics({
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        })

        predictions_df = pd.DataFrame({
            'true_labels': test_labels_tta.tolist(),
            'predictions': test_predictions_tta.tolist(),
            'probabilities': test_probs_tta.tolist(),
            'certainties_s': test_certainties_s_tta.cpu().tolist(),
            'confidences_s': test_confidences_s_tta.cpu().tolist()
        })
        predictions_df.to_csv(f"{self.config.model.name}_predictions.csv", index=False)

        print('Performing visualization...')

        cm = confusion_matrix(test_labels_tta, test_predictions_tta)
        cm_display = ConfusionMatrixDisplay(cm, display_labels=self.config.dataset.class_names).plot()
        cm_display.ax_.set_title(f'TTA predictions {self.config.model.name}_{self.config.dataset.seed}_{self.config.dataset.bagging_size}')
        plt.savefig(f"{self.config.model.name}_{self.config.dataset.seed}_confusion_matrix.png")

        test_attr_tta = test_vis_tta(self,
                                    self.trainer.datamodule.test_dataloader(),
                                    num_classes=self.config.model.num_classes,
                                    loss_fun=self.config.model.loss_fun,
                                    seed=self.config.dataset.seed,
                                    figs=1,
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

        name = f'{self.config.model.name}_{self.config.dataset.seed}_{self.config.dataset.bagging_size}'
        hist(df, 'Mode_confidence_(soft)', (4, 4), name=f'{name} Mode Confidence (soft)')
        hist(df, 'Mode_certainty_(soft)', (4, 4), name=f'{name} Mode Certainty (soft)')

        cm = confusion_matrix(mode_labels.cpu(), mode_predictions.cpu())
        print('Mode based TTA predictions (TTAM)')
        cm_display = ConfusionMatrixDisplay(cm, display_labels=self.config.dataset.class_names).plot()
        cm_display.ax_.set_title(f'Mode based TTA predictions (TTAM) {self.config.model.name}_{self.config.dataset.seed}_{self.config.dataset.bagging_size}')
        plt.savefig(f"{self.config.model.name}_{self.config.dataset.seed}_mode_confusion_matrix.png")

        print('Calculating weighted predictions...')

        weightedPred = ttaWeightedPred(test_attr_tta['labels_tta'], test_attr_tta['predictions_tta'], test_attr_tta['confidences_s_tta'], test_attr_tta['certainties_s_tta'], class_names=self.config.dataset.class_names)
        n_correct_TTAWCo_S = (weightedPred['predictionsCo'] == mode_labels).sum().item()
        n_correct_TTAWCe_S = (weightedPred['predictionsCe'] == mode_labels).sum().item()

        n_samples = mode_labels.shape[0]
        accuracy_TTAWCo_S = n_correct_TTAWCo_S / n_samples
        accuracy_TTAWCe_S = n_correct_TTAWCe_S / n_samples

        print('---------------------- Soft TTA predictions -----------------------------')
        print('n_samples', n_samples)
        print(f'accuracy of confidences based soft TTA predictions = {accuracy_TTAWCo_S}')
        print(f'accuracy of certainties based soft TTA predictions = {accuracy_TTAWCe_S}')

        cm = confusion_matrix(mode_labels.cpu(), weightedPred['predictionsCo'].cpu())
        print('Confidence based soft TTA predictions (TTAWCo-S)')
        cm_display = ConfusionMatrixDisplay(cm, display_labels=self.config.dataset.class_names).plot()
        cm_display.ax_.set_title(f'Confidence based soft TTA predictions (TTAWCo-S) {self.config.model.name}_{self.config.dataset.seed}_{self.config.dataset.bagging_size}')
        plt.savefig(f"{self.config.model.name}_{self.config.dataset.seed}_confidence_confusion_matrix.png")

        cm = confusion_matrix(mode_labels.cpu(), weightedPred['predictionsCe'].cpu())
        print('Certainty based soft TTA predictions (TTAWCe-S)')
        cm_display = ConfusionMatrixDisplay(cm, display_labels=self.config.dataset.class_names).plot()
        cm_display.ax_.set_title(f'Certainty based soft TTA predictions (TTAWCe-S) {self.config.model.name}_{self.config.dataset.seed}_{self.config.dataset.bagging_size}')
        plt.savefig(f"{self.config.model.name}_{self.config.dataset.seed}_certainty_confusion_matrix.png")

        # Summary table
        test_accuracy_summary = {
            'ID': self.logger._task.id,
            'dataset': self.config.dataset.name,
            'loss_fun': self.config.model.loss_fun,
            'margin': self.config.model.margin,
            'weight_decay': self.config.model.optimizer_hparams.weight_decay,
            'label_smoothing': self.config.model.label_smoothing,
            'bagging_size': self.config.dataset.bagging_size,
            'seed': self.config.dataset.seed,
            '# samples': len(test_labels_tta),
            '# TTA': self.config.dataset.num_tta,
            'F1 (without TTA)': f1_no_tta,
            'ROCAUC (without TTA)': roc_auc_no_tta,
            'Acc (without TTA)': accuracy_no_tta,
            'F1 TTA': f1,
            'ROCAUC': roc_auc,
            'Acc TTAM': accuracy_tta(mode_labels, mode_predictions)['acc_without_u'],
            'Acc TTAWCo-S': accuracy_TTAWCo_S,
            'Acc TTAWCe-S': accuracy_TTAWCe_S,
            'ECE': calculate_ece(test_probs_tta, test_labels_tta, num_classes=self.config.model.num_classes)
        }

        summary_df = pd.DataFrame([test_accuracy_summary])
        summary_df.to_csv(f"{self.config.model.name}_{self.config.dataset.seed}_summary.csv", index=False)